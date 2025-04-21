// # Define TORCH_COMPILE macro

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>
#include <stdio.h>

#define CLAMP(value, min, max) ((value) < (min) ? (min) : ((value) > (max) ? (max) : (value)))
#define ABS(x) ((x) < 0 ? -(x) : (x))

using namespace kittens;
namespace cg = cooperative_groups;

template<int T, int H, int W, int D, int TEXT> 
struct fwd_attend_ker_metadata {
    constexpr static int tile_size = T*H*W;
    constexpr static int tile_width = (D);
    constexpr static int qo_height  = (64);
    constexpr static int kv_height  = (64);
    constexpr static int n_kv_per_tile = tile_size / kv_height; 
    constexpr static int n_qo_per_tile = tile_size / qo_height;


    constexpr static int stages     = (3); 

    constexpr static int max_text_len = TEXT;
    constexpr static int text_kv_blocks = max_text_len / kv_height;

    static_assert(tile_size % qo_height == 0, "tile_size T*H*W must be divisible by qo_height");
    constexpr static int CONSUMER_WARPGROUPS = (tile_size / qo_height); 
    constexpr static int PRODUCER_WARPGROUPS = (1); 
    constexpr static int NUM_WARPGROUPS      = (CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS); 
    constexpr static int NUM_WORKERS         = (NUM_WARPGROUPS*kittens::WARPGROUP_WARPS); 

};



template<int T, int H, int W, int D, int TEXT> 
struct fwd_globals {
    using K = fwd_attend_ker_metadata<T, H, W, D, TEXT>;

    using q_tile    =         st_bf<K::qo_height, K::tile_width>;
    using k_tile    =         st_bf<K::kv_height, K::tile_width>;
    using v_tile    =         st_bf<K::kv_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using o_tile    =         st_bf<K::qo_height, K::tile_width>;

    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;

    const int N;  // Total seqlen (image + text)
    const int text_L;  // Actual text seqlen
    const int hr;  // Head ratio

    // DT: attended tiles in time
    // DH: attended tiles in height
    // DW: attended tiles in width
    // CT: total number of tiles in time
    // CH: total number of tiles in height
    // CW: total number of tiles in width
    const int DT, DH, DW, CT, CH, CW;
    const bool text_q, text_kv;  // text_q: Whether process text as query
// text_kv: Whether process text as key and value
};

// H: tile size of height
// W: tile size of width
// T: tile size of time
// D: head dim
// TEXT: max text length
template<int H, int W, int T, int D, int TEXT>
__global__  __launch_bounds__((fwd_attend_ker_metadata<T, H, W, D, TEXT>::NUM_WORKERS)*kittens::WARP_THREADS, 1)
void fwd_attend_ker(
    const __grid_constant__ fwd_globals<T, H, W, D, TEXT> g) 
{

    // static_assert(is_causal == false, "Currently only supports non-causal attention");
    
    // if(blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("text_L: %d\n", g.text_L);
    // }

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = fwd_attend_ker_metadata<T, H, W, D, TEXT>;
    using globals = fwd_globals<T, H, W, D, TEXT> ;

    using q_tile = typename globals::q_tile;
    using k_tile = typename globals::k_tile;
    using v_tile = typename globals::v_tile;
    using l_col_vec = typename globals::l_col_vec;
    using o_tile = typename globals::o_tile;

    q_tile    (&q_smem)[K::CONSUMER_WARPGROUPS] = al.allocate<q_tile, K::CONSUMER_WARPGROUPS>();
    k_tile    (&k_smem)[K::stages]           = al.allocate<k_tile, K::stages          >();
    v_tile    (&v_smem)[K::stages]           = al.allocate<v_tile, K::stages          >();
    l_col_vec (&l_smem)[K::CONSUMER_WARPGROUPS] = al.allocate<l_col_vec, K::CONSUMER_WARPGROUPS>();
    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem);

    int img_kv_blocks;
    int kv_blocks   = g.N / (K::kv_height);
    if (g.text_kv) {
        img_kv_blocks = kv_blocks - K::text_kv_blocks;  // todo1
    } else {
        img_kv_blocks = kv_blocks;
    }
    int kv_head_idx = blockIdx.y / g.hr;
    int seq_idx;
    if (g.text_q) {
        seq_idx = g.CT * g.CH * g.CW * K::n_qo_per_tile + blockIdx.x * K::CONSUMER_WARPGROUPS;  // todo 6
    } else {
        seq_idx = blockIdx.x * K::CONSUMER_WARPGROUPS; 
    }
    __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived[K::stages], v_smem_arrived[K::stages], compute_done[K::stages];
    if (threadIdx.x == 0) { 
        init_semaphore(qsmem_semaphore, 0, 1); 
        for(int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1); 
            init_semaphore(v_smem_arrived[j], 0, 1); 
            init_semaphore(compute_done[j], K::CONSUMER_WARPGROUPS, 0); 
        }

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));

        for (int wg = 0; wg < K::CONSUMER_WARPGROUPS; wg++) {
            coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_tile_idx, qsmem_semaphore);
        }

        if (g.text_q) {
            for (int j = 0; j < K::stages - 1; j++) {
                coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, j, 0};
                tma::expect_bytes(k_smem_arrived[j], sizeof(k_tile));
                tma::load_async(k_smem[j], g.k, kv_tile_idx, k_smem_arrived[j]);
                tma::expect_bytes(v_smem_arrived[j], sizeof(v_tile));
                tma::load_async(v_smem[j], g.v, kv_tile_idx, v_smem_arrived[j]);
            }
        } else {
            int qt = seq_idx / K::n_qo_per_tile / (g.CH * g.CW);
            int qh = (seq_idx / K::n_qo_per_tile) % (g.CH * g.CW) / g.CW;
            int qw = (seq_idx / K::n_qo_per_tile) % g.CW;
            qt = CLAMP(qt, g.DT, g.CT-g.DT-1);
            qh = CLAMP(qh, g.DH, g.CH-g.DH-1);
            qw = CLAMP(qw, g.DW, g.CW-g.DW-1);
            int count = 0;
            int j = 0;
            while (count < K::stages - 1) {
                int kt = j / K::n_kv_per_tile / (g.CH * g.CW);
                int kh = (j / K::n_kv_per_tile) % (g.CH * g.CW) / g.CW;
                int kw = (j / K::n_kv_per_tile) % g.CW;
                bool mask = (ABS(qt - kt) <= g.DT) && (ABS(qh - kh) <= g.DH) && (ABS(qw - kw) <= g.DW);
                if (mask){
                    coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, j, 0};
                    tma::expect_bytes(k_smem_arrived[count], sizeof(k_tile));
                    tma::load_async(k_smem[count], g.k, kv_tile_idx, k_smem_arrived[count]);
                    tma::expect_bytes(v_smem_arrived[count], sizeof(v_tile));
                    tma::load_async(v_smem[count], g.v, kv_tile_idx, v_smem_arrived[count]);
                    count += 1;
                }
                j += 1;
            }
        }
    }
    __syncthreads(); 

    int pipe_idx = K::stages - 1; 
    
    // Producer warpgroup
    if(warpgroupid == K::NUM_WARPGROUPS-1) {
        warpgroup::decrease_registers<32>();      
        
        int kv_iters; 
        // if constexpr (is_causal) {
        //     kv_iters = (seq_idx * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)) - 1 + (CONSUMER_WARPGROUPS * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)); 
        //     kv_iters = ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) == 0) ? (0) : ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) - 1);
        // }
        //else 
        { kv_iters = kv_blocks - (K::stages-1);}  // todo2 

        if(warpid == K::NUM_WORKERS-kittens::WARPGROUP_WARPS) {  // leading warp of the producer warpgroup
            if (g.text_q) {
                for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                    coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_idx + 1, 0};
                    tma::expect_bytes(k_smem_arrived[(kv_idx+1)%K::stages], sizeof(k_tile));
                    tma::load_async(k_smem[(kv_idx+1)%K::stages], g.k, kv_tile_idx, k_smem_arrived[(kv_idx+1)%K::stages]);
                    tma::expect_bytes(v_smem_arrived[(kv_idx+1)%K::stages], sizeof(v_tile));
                    tma::load_async(v_smem[(kv_idx+1)%K::stages], g.v, kv_tile_idx, v_smem_arrived[(kv_idx+1)%K::stages]);
                        kittens::wait(compute_done[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
                }
            } else {
                int qt = seq_idx / K::n_qo_per_tile / (g.CH * g.CW);
                int qh = (seq_idx / K::n_qo_per_tile) % (g.CH * g.CW) / g.CW;
                int qw = (seq_idx / K::n_qo_per_tile) % g.CW;
                qt = CLAMP(qt, g.DT, g.CT-g.DT-1);
                qh = CLAMP(qh, g.DH, g.CH-g.DH-1);
                qw = CLAMP(qw, g.DW, g.CW-g.DW-1);
                int k_t_min = CLAMP(qt-g.DT, 0, g.CT-1);
                int k_t_max = CLAMP(qt+g.DT, 0, g.CT-1);
                int k_h_min = CLAMP(qh-g.DH, 0, g.CH-1);
                int k_h_max = CLAMP(qh+g.DH, 0, g.CH-1);
                int k_w_min = CLAMP(qw-g.DW, 0, g.CW-1);
                int k_w_max = CLAMP(qw+g.DW, 0, g.CW-1);
                int count = 0;
                for (int kt = k_t_min; kt <= k_t_max; kt++) {
                    for (int kh = k_h_min; kh <= k_h_max; kh++) {
                        for (int kw = k_w_min; kw <= k_w_max; kw++) {
                            for (int j = 0; j < K::n_kv_per_tile; j++){
                                if (count >= K::stages - 1) {
                                    int index = ((kt * (g.CH * g.CW)) + (kh * g.CW) + kw) * K::n_kv_per_tile + j;
                                    coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, index, 0};
                                    tma::expect_bytes(k_smem_arrived[count%K::stages], sizeof(k_tile));
                                    tma::load_async(k_smem[count%K::stages], g.k, kv_tile_idx, k_smem_arrived[count%K::stages]);
                                    tma::expect_bytes(v_smem_arrived[count%K::stages], sizeof(v_tile));
                                    tma::load_async(v_smem[count%K::stages], g.v, kv_tile_idx, v_smem_arrived[count%K::stages]);
                                    kittens::wait(compute_done[(count - 1)%K::stages], ((count - 1)/K::stages)%2);
                                    count += 1;
                                } else {
                                    count += 1;
                                }
                            }
                        }
                    }
                }
                // for text 
                for (int index = img_kv_blocks; index < kv_blocks; index++) {
                    coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, index, 0};
                    tma::expect_bytes(k_smem_arrived[count%K::stages], sizeof(k_tile));
                    tma::load_async(k_smem[count%K::stages], g.k, kv_tile_idx, k_smem_arrived[count%K::stages]);
                    tma::expect_bytes(v_smem_arrived[count%K::stages], sizeof(v_tile));
                    tma::load_async(v_smem[count%K::stages], g.v, kv_tile_idx, v_smem_arrived[count%K::stages]);
                    kittens::wait(compute_done[(count - 1)%K::stages], ((count - 1)/K::stages)%2);
                    count += 1;
                }
            }


        }
    }
    // Consumer warpgroup
    else {
        warpgroup::increase_registers<160>();

        rt_fl<16, K::kv_height>  att_block;
        rt_bf<16, K::kv_height>  att_block_mma;
        rt_fl<16, K::tile_width> o_reg;
        
        col_vec<rt_fl<16, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;
        
        neg_infty(max_vec);
        zero(norm_vec);
        zero(o_reg);

        int kv_iters; 
        // if constexpr (is_causal) {
        //     kv_iters = (seq_idx * 4) - 1 + (CONSUMER_WARPGROUPS * 4);
        //     kv_iters = (kv_iters/8);
        // }
        // else 
        if (g.text_q){ 
            // the last three kv blocks are for text, we process them separately
            kv_iters = img_kv_blocks - 1;
        } else {
            kv_iters = CLAMP(g.DT*2+1, 1, g.CT) * CLAMP(g.DH*2+1, 1, g.CH) * CLAMP(g.DW*2+1, 1, g.CW) * K::n_kv_per_tile - 1 ; 
        }

        kittens::wait(qsmem_semaphore, 0);
        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {

            kittens::wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[(kv_idx)%K::stages]);
            
            copy(max_vec_last_scaled, max_vec);
            if constexpr (D == 64) { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f); }
            else                   { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f); }
            
            warpgroup::mma_async_wait();

            row_max(max_vec, att_block, max_vec);
            
            if constexpr (D == 64) { 
                mul(att_block, att_block,    1.44269504089f*0.125f); 
                mul(max_vec_scaled, max_vec, 1.44269504089f*0.125f);
            }
            else                   { 
                mul(att_block, att_block,    1.44269504089f*0.08838834764f); 
                mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
            }

            sub_row(att_block, att_block, max_vec_scaled);
            exp2(att_block, att_block);
            sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            exp2(max_vec_last_scaled,       max_vec_last_scaled);
            mul(norm_vec,            norm_vec,     max_vec_last_scaled);
            row_sum(norm_vec,  att_block, norm_vec);
            add(att_block, att_block, 0.f);
            copy(att_block_mma, att_block); 
            mul_row(o_reg, o_reg, max_vec_last_scaled); 

            kittens::wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2); 

            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[(kv_idx)%K::stages]);
            warpgroup::mma_async_wait();

            if(warpgroup::laneid() == 0) arrive(compute_done[(kv_idx)%K::stages], 1);
        }
        // the last three kv blocks are for text, we process them separately
        if (g.text_kv) {
            for (auto kv_idx = kv_iters + 1; kv_idx <= kv_iters + 3; kv_idx++) {  // todo 8

                kittens::wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
                warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[(kv_idx)%K::stages]);
                
                copy(max_vec_last_scaled, max_vec);
                if constexpr (D == 64) { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f); }
                else                   { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f); }
                
                warpgroup::mma_async_wait();
                // apply non-pad mask
                int offset = g.text_L - (kv_idx - (kv_iters + 1)) * K::kv_height;
                // printf("k_idx_start: %d, k_idx_end: %d, text_end: %d, offset: %d\n", k_idx_start, k_idx_end, text_end, offset);
                right_fill(att_block, att_block, offset, base_types::constants<float>::neg_infty());


                row_max(max_vec, att_block, max_vec);
                
                if constexpr (D == 64) { 
                    mul(att_block, att_block,    1.44269504089f*0.125f); 
                    mul(max_vec_scaled, max_vec, 1.44269504089f*0.125f);
                }
                else                   { 
                    mul(att_block, att_block,    1.44269504089f*0.08838834764f); 
                    mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
                }

                sub_row(att_block, att_block, max_vec_scaled);
                exp2(att_block, att_block);
                sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
                exp2(max_vec_last_scaled,       max_vec_last_scaled);
                mul(norm_vec,            norm_vec,     max_vec_last_scaled);
                row_sum(norm_vec,  att_block, norm_vec);
                add(att_block, att_block, 0.f);
                copy(att_block_mma, att_block); 
                mul_row(o_reg, o_reg, max_vec_last_scaled); 

                kittens::wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2); 

                warpgroup::mma_AB(o_reg, att_block_mma, v_smem[(kv_idx)%K::stages]);
                warpgroup::mma_async_wait();

                if(warpgroup::laneid() == 0) arrive(compute_done[(kv_idx)%K::stages], 1);
            }
        }

        div_row(o_reg, o_reg, norm_vec);
        warpgroup::store(o_smem[warpgroupid], o_reg); 
        warpgroup::sync(warpgroupid+4);

        if (warpid % 4 == 0) {
            coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + warpgroupid, 0};
            tma::store_async(g.o, o_smem[warpgroupid], o_tile_idx);
        }

        mul(max_vec_scaled,   max_vec_scaled, 0.69314718056f);
        log(norm_vec, norm_vec);
        add(norm_vec, norm_vec, max_vec_scaled);

        if constexpr (D == 64) { mul(norm_vec, norm_vec, -8.0f); }
        else                   { mul(norm_vec, norm_vec, -11.313708499f); }
    
        warpgroup::store(l_smem[warpgroupid], norm_vec);
        warpgroup::sync(warpgroupid+4);

        if (warpid % 4 == 0) {
            coord<l_col_vec> tile_idx = {blockIdx.z, blockIdx.y, 0, (seq_idx) + warpgroupid};
            tma::store_async(g.l, l_smem[warpgroupid], tile_idx);
        }
        tma::store_async_wait();
    }
}



#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>



torch::Tensor 
sta_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor img_size, torch::Tensor tile_size, torch::Tensor kernel_size, int text_length, bool process_text)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    auto batch    = q.size(0);
    auto seq_len  = q.size(2); 
    auto head_dim = q.size(3);  
    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(q.size(0) == batch, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0) == batch, "K batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0) == batch, "V batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(2) == seq_len, "Q sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(k.size(2) == seq_len, "K sequence length dimension - idx 2 - must match for all inputs");
    TORCH_CHECK(v.size(2) == seq_len, "V sequence length dimension - idx 2 - must match for all inputs");

    TORCH_CHECK(q.size(3) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3) == head_dim, "K head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3) == head_dim, "V head dimension - idx 3 - must match for all non-vector inputs");

    TORCH_CHECK(qo_heads >= kv_heads, "QO heads must be greater than or equal to KV heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "QO heads must be divisible by KV heads");
    TORCH_CHECK(q.size(1) == qo_heads, "QO head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");  

    auto hr = qo_heads / kv_heads;

    c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr = v.data_ptr<c10::BFloat16>();

    bf16*  d_q = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k = reinterpret_cast<bf16*>(k_ptr);
    bf16*  d_v = reinterpret_cast<bf16*>(v_ptr);
    

    torch::Tensor l_vec = torch::empty({static_cast<const uint>(batch), 
                                        static_cast<const uint>(qo_heads), 
                                        static_cast<const uint>(seq_len), 
                                        static_cast<const uint>(1)}, 
                                        torch::TensorOptions().dtype(torch::kFloat).device(q.device()).memory_format(at::MemoryFormat::Contiguous));
        

    bf16*  o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16*  d_o   = reinterpret_cast<bf16*>(o_ptr);

    float* l_ptr = reinterpret_cast<float*>(l_vec.data_ptr<float>());
    float* d_l   = reinterpret_cast<float*>(l_ptr);

    cudaDeviceSynchronize();  // TODO: Is this necessary?
    auto stream = at::cuda::getCurrentCUDAStream().stream(); 


    TORCH_CHECK(head_dim == 128, "head_dim must be 128");  // TODO: Should be able to relax this
    int D = head_dim;
    

    TORCH_CHECK(img_size.size(0) == 3, "img_size must be 3D");
    TORCH_CHECK(tile_size.size(0) == 3, "tile_size must be 3D");
    TORCH_CHECK(kernel_size.size(0) == 3, "kernel_size must be 3D");
    

    int T = tile_size[0].item<int>();
    int H = tile_size[1].item<int>();
    int W = tile_size[2].item<int>();

    int img_size_T = img_size[0].item<int>();
    int img_size_H = img_size[1].item<int>();
    int img_size_W = img_size[2].item<int>();

    TORCH_CHECK(img_size_T % T == 0, "img_size must be divisible by tile_size in the time dimension");
    TORCH_CHECK(img_size_H % H == 0, "img_size must be divisible by tile_size in the height dimension");
    TORCH_CHECK(img_size_W % W == 0, "img_size must be divisible by tile_size in the width dimension");

    int CT = img_size_T / T;
    int CH = img_size_H / H;
    int CW = img_size_W / W;

    int kernel_size_T = kernel_size[0].item<int>();
    int kernel_size_H = kernel_size[1].item<int>();
    int kernel_size_W = kernel_size[2].item<int>();

    TORCH_CHECK(kernel_size_T <= CT, "kernel_size must be less than or equal to num_tiles in the time dimension");
    TORCH_CHECK(kernel_size_H <= CH, "kernel_size must be less than or equal to num_tiles in the height dimension");
    TORCH_CHECK(kernel_size_W <= CW, "kernel_size must be less than or equal to num_tiles in the width dimension");

    // size -> distance
    int DT = kernel_size_T / 2;
    int DH = kernel_size_H / 2;
    int DW = kernel_size_W / 2;


    int img_len = img_size_T * img_size_H * img_size_W;
    int TEXT = seq_len - img_len;

    TORCH_CHECK(TEXT >= text_length, "max_text_len must be no less than than text_length");


    
    bool dispatched = false;


    #define MAYBE_DISPATCH_TILE_SIZE(argT, argH, argW, argD, argTEXT) \
    if (T == argT && H == argH && W == argW && D == argD && TEXT == argTEXT) { \
        using K = fwd_attend_ker_metadata<argT, argH, argW, argD, argTEXT>; \
        using globals = fwd_globals<argT, argH, argW, argD, argTEXT>; \
        TORCH_CHECK(img_len % (K::CONSUMER_WARPGROUPS * K::qo_height) == 0, "Sequence length of the image part must be divisible by 64"); \
        TORCH_CHECK(TEXT % (K::CONSUMER_WARPGROUPS * K::qo_height) == 0, "Sequence length of the text part must be divisible by 64"); \
        globals::q_gl qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), K::tile_width}; \
        globals::k_gl kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), K::tile_width}; \
        globals::v_gl vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), K::tile_width}; \
        globals::l_gl lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(seq_len)}; \
        globals::o_gl og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), K::tile_width}; \
        globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, static_cast<int>(seq_len),  static_cast<int>(text_length), static_cast<int>(hr), DT, DH, DW, CT, CH, CW, /*text_q*/process_text, /*text_kv*/true}; \
        dim3 grid_dim; \
        if (!process_text) { \
            grid_dim = dim3(img_len / (K::CONSUMER_WARPGROUPS * K::qo_height), qo_heads, batch); \
        } else { \
            grid_dim = dim3(TEXT / (K::CONSUMER_WARPGROUPS * K::qo_height), qo_heads, batch); \
        } \
        auto threads = K::NUM_WORKERS * kittens::WARP_THREADS; \
        auto mem_size = kittens::MAX_SHARED_MEMORY; \
        cudaFuncSetAttribute(fwd_attend_ker<argH, argW, argT, argD, argTEXT>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size); \
        fwd_attend_ker<argH, argW, argT, argD, argTEXT><<<grid_dim, threads, mem_size, stream>>>(g); \
        dispatched = true; \
    }


    // Note begin: Add more supported tile_size here
    MAYBE_DISPATCH_TILE_SIZE(8, 4, 4, 128, 256);

    // Note end


    if (!dispatched) {
        TORCH_CHECK(false, "Unsupported: T:", T, " H:", H, " W:", W, " D:", D, " TEXT:", TEXT);
    }


    return o;

    #undef MAYBE_DISPATCH_TILE_SIZE
}
