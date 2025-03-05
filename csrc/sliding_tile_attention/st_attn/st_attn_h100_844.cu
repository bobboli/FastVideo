// # Define TORCH_COMPILE macro

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>
#include <stdio.h>

#define CLAMP(value, min, max) ((value) < (min) ? (min) : ((value) > (max) ? (max) : (value)))
#define ABS(x) ((x) < 0 ? -(x) : (x))

constexpr int CONSUMER_WARPGROUPS = (2); 
constexpr int PRODUCER_WARPGROUPS = (1); 
constexpr int NUM_WARPGROUPS      = (CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS); 
constexpr int NUM_WORKERS         = (NUM_WARPGROUPS*kittens::WARPGROUP_WARPS); 

using namespace kittens;
namespace cg = cooperative_groups;

template<int D> struct fwd_attend_ker_tile_dims {};
template<> struct fwd_attend_ker_tile_dims<64> {
    constexpr static int tile_width = (64);
    constexpr static int qo_height  = (64);
    constexpr static int kv_height  = (64);
    constexpr static int stages     = (3); 
};
template<> struct fwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height  = (64);
    constexpr static int kv_height  = (64);
    constexpr static int stages     = (3); 
};

constexpr int max_text_len = 256;


template<int D> struct fwd_globals {
    using q_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using k_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using v_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::kv_height, fwd_attend_ker_tile_dims<D>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>>;
    using o_tile    =         st_bf<fwd_attend_ker_tile_dims<D>::qo_height, fwd_attend_ker_tile_dims<D>::tile_width>;

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

    const int N; 
    const int text_L;
    const int hr;
};


template<int D, bool is_causal, bool text_q, bool text_kv>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void fwd_attend_ker(const __grid_constant__ fwd_globals<D> g, 
    const int DT, const int DH, const int DW, 
    const int CT, const int CH, const int CW) 
{

    static_assert(is_causal == false, "Currently only supports non-causal attention");

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = fwd_attend_ker_tile_dims<D>;
    constexpr int text_kv_blocks = max_text_len / K::kv_height;  // 4

    // tile_size = 8,4,4 = 128    qo_height * CONSUMER_WARPGROUPS=128, kv_height=64
    constexpr int tile_size = 8*4*4;
    constexpr int n_kv_per_attn_tile = tile_size / K::kv_height;  // 2
    constexpr int n_qo_per_attn_tile = tile_size / K::qo_height;  // 2

    // if(blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("text_L: %d\n", g.text_L);
    // }

    using q_tile    =         st_bf<K::qo_height, K::tile_width>;
    using k_tile    =         st_bf<K::kv_height, K::tile_width>;
    using v_tile    =         st_bf<K::kv_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using o_tile    =         st_bf<K::qo_height, K::tile_width>;
    
    q_tile    (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    k_tile    (&k_smem)[K::stages]           = al.allocate<k_tile, K::stages          >();
    v_tile    (&v_smem)[K::stages]           = al.allocate<v_tile, K::stages          >();
    l_col_vec (&l_smem)[CONSUMER_WARPGROUPS] = al.allocate<l_col_vec, CONSUMER_WARPGROUPS>();
    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem);
    int img_kv_blocks;
    int kv_blocks   = g.N / (K::kv_height);
    if constexpr (text_kv) {
        img_kv_blocks = kv_blocks - text_kv_blocks;  // todo1
    } else {
        img_kv_blocks = kv_blocks;
    }
    int kv_head_idx = blockIdx.y / g.hr;
    int seq_idx;
    if constexpr (text_q) {
        seq_idx = CT * CH * CW * n_qo_per_attn_tile + blockIdx.x * CONSUMER_WARPGROUPS;  // todo 6
    } else {
        seq_idx = blockIdx.x * CONSUMER_WARPGROUPS; 
    }
    __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived[K::stages], v_smem_arrived[K::stages], compute_done[K::stages];
    if (threadIdx.x == 0) { 
        init_semaphore(qsmem_semaphore, 0, 1); 
        for(int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1); 
            init_semaphore(v_smem_arrived[j], 0, 1); 
            init_semaphore(compute_done[j], CONSUMER_WARPGROUPS, 0); 
        }

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));

        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_tile_idx, qsmem_semaphore);
        }

        if constexpr (text_q){  // todo7
            for (int j = 0; j < K::stages - 1; j++) {
                coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, j, 0};
                tma::expect_bytes(k_smem_arrived[j], sizeof(k_tile));
                tma::load_async(k_smem[j], g.k, kv_tile_idx, k_smem_arrived[j]);
                tma::expect_bytes(v_smem_arrived[j], sizeof(v_tile));
                tma::load_async(v_smem[j], g.v, kv_tile_idx, v_smem_arrived[j]);
            }
        } else {
            int qt = seq_idx / n_qo_per_attn_tile / (CH * CW);
            int qh = (seq_idx / n_qo_per_attn_tile) % (CH * CW) / CW;
            int qw = (seq_idx / n_qo_per_attn_tile) % CW;
            qt = CLAMP(qt, DT, CT-DT-1);
            qh = CLAMP(qh, DH, CH-DH-1);
            qw = CLAMP(qw, DW, CW-DW-1);
            int count = 0;
            int j = 0;
            while (count < K::stages - 1) {
                int kt = j / n_kv_per_attn_tile / (CH * CW);
                int kh = (j / n_kv_per_attn_tile) % (CH * CW) / CW;
                int kw = (j / n_kv_per_attn_tile) % CW;
                bool mask = (ABS(qt - kt) <= DT) && (ABS(qh - kh) <= DH) && (ABS(qw - kw) <= DW);
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
    if(warpgroupid == NUM_WARPGROUPS-1) {
        warpgroup::decrease_registers<32>();      
        
        int kv_iters; 
        // if constexpr (is_causal) {
        //     kv_iters = (seq_idx * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)) - 1 + (CONSUMER_WARPGROUPS * (K::qo_height/kittens::TILE_ROW_DIM<bf16>)); 
        //     kv_iters = ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) == 0) ? (0) : ((kv_iters / (K::kv_height/kittens::TILE_ROW_DIM<bf16>)) - 1);
        // }
        //else 
        { kv_iters = kv_blocks - (K::stages-1);}  // todo2 

        if(warpid == NUM_WORKERS-4) {  // leading warp of the producer warpgroup
            if constexpr (text_q){
                for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                    coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_idx + 1, 0};
                    tma::expect_bytes(k_smem_arrived[(kv_idx+1)%K::stages], sizeof(k_tile));
                    tma::load_async(k_smem[(kv_idx+1)%K::stages], g.k, kv_tile_idx, k_smem_arrived[(kv_idx+1)%K::stages]);
                    tma::expect_bytes(v_smem_arrived[(kv_idx+1)%K::stages], sizeof(v_tile));
                    tma::load_async(v_smem[(kv_idx+1)%K::stages], g.v, kv_tile_idx, v_smem_arrived[(kv_idx+1)%K::stages]);
                    kittens::wait(compute_done[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
                }
            } else {
                int qt = seq_idx / n_qo_per_attn_tile / (CH * CW);
                int qh = (seq_idx / n_qo_per_attn_tile) % (CH * CW) / CW;
                int qw = (seq_idx / n_qo_per_attn_tile) % CW;
                qt = CLAMP(qt, DT, CT-DT-1);
                qh = CLAMP(qh, DH, CH-DH-1);
                qw = CLAMP(qw, DW, CW-DW-1);
                int k_t_min = CLAMP(qt-DT, 0, CT-1);
                int k_t_max = CLAMP(qt+DT, 0, CT-1);
                int k_h_min = CLAMP(qh-DH, 0, CH-1);
                int k_h_max = CLAMP(qh+DH, 0, CH-1);
                int k_w_min = CLAMP(qw-DW, 0, CW-1);
                int k_w_max = CLAMP(qw+DW, 0, CW-1);
                int count = 0;
                for (int kt = k_t_min; kt <= k_t_max; kt++) {
                    for (int kh = k_h_min; kh <= k_h_max; kh++) {
                        for (int kw = k_w_min; kw <= k_w_max; kw++) {
                            for (int j = 0; j <= 2; j++){
                                if (count >= K::stages - 1) {
                                    int index = ((kt * (CH * CW)) + (kh * CW) + kw) * 3 + j;
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
        if constexpr (text_q){ 
            // the last three kv blocks are for text, we process them separately
            kv_iters = img_kv_blocks - 1;
        } else {
            kv_iters = CLAMP(DT*2+1, 1, CT) * CLAMP(DH*2+1, 1, CH) * CLAMP(DW*2+1, 1, CW) * 3 - 1 ; 
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
        if constexpr(text_kv) {
            for (auto kv_idx = kv_iters + 1; kv_idx <= kv_iters + 3; kv_idx++) {

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
sta_forward_844(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, int kernel_t_size, int kernel_h_size, int kernel_w_size, int text_length, bool process_text, bool has_text)
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

    cudaDeviceSynchronize();
    auto stream = at::cuda::getCurrentCUDAStream().stream(); 


    TORCH_CHECK(head_dim == 128, "head_dim must be 128");
    

    
    using q_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>;
    using k_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width>;
    using v_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::kv_height, fwd_attend_ker_tile_dims<128>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>>;
    using o_tile    =         st_bf<fwd_attend_ker_tile_dims<128>::qo_height, fwd_attend_ker_tile_dims<128>::tile_width>;

    using q_global = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_global = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_global = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_global = gl<bf16,  -1, -1, -1, -1, o_tile>;

    using globals      = fwd_globals<128>;
    using K = fwd_attend_ker_tile_dims<128>;

    q_global qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};
    k_global kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
    v_global vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(seq_len), 128U};
    l_global lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(seq_len)};
    o_global og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(seq_len), 128U};

    globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, static_cast<int>(seq_len),  static_cast<int>(text_length), static_cast<int>(hr)};

    auto mem_size = kittens::MAX_SHARED_MEMORY;
    auto threads  = NUM_WORKERS * kittens::WARP_THREADS;

    bool is_supported_kernel = false;
    // latent_size = (16, 32, 56)
    // tile_size = (8,4,4)
    // n_tiles = (2,8,14)

    const int supported_kernels[][3] = {
        {1, 1, 1},
        {1, 3, 3},
        {1, 3, 7},
        {1, 5, 5},
        {1, 5,7},
        {1, 5, 11},
        {1, 8, 11},
        {1, 8, 14},
        {2, 1, 1},
        {2, 3, 3},
        {2, 3, 7},
        {2, 5, 5},
        {2, 5, 7},
        {2, 5, 11},
        {2, 8, 11},
        {2, 8, 14}
    };

    const int num_kernels = sizeof(supported_kernels) / sizeof(supported_kernels[0]);
    for (int i = 0; i < num_kernels; i++) {
        if (kernel_t_size == supported_kernels[i][0] && 
            kernel_h_size == supported_kernels[i][1] && 
            kernel_w_size == supported_kernels[i][2]) {
            is_supported_kernel = true;
            break;
        }
    }
    
    TORCH_CHECK(is_supported_kernel, "Unsupported kernel dimensions: ", 
                kernel_t_size, "x", kernel_h_size, "x", kernel_w_size);

    int DT = kernel_t_size / 2;
    int DH = kernel_h_size / 2;
    int DW = kernel_w_size / 2;

    constexpr int CT = 2;
    constexpr int CH = 8;
    constexpr int CW = 14;
    
    TORCH_CHECK(has_text, "Currently only supports has_text");
    TORCH_CHECK( (seq_len - max_text_len) % (CONSUMER_WARPGROUPS * K::qo_height) == 0, "sequence length of the image part must be divisible by 128");
    TORCH_CHECK( max_text_len % (CONSUMER_WARPGROUPS * K::qo_height) == 0, "sequence length of the text part must be divisible by 128");
    dim3 grid_image((seq_len - max_text_len)/(CONSUMER_WARPGROUPS * K::qo_height * 4), qo_heads, batch);  //todo4
    dim3 grid_text(max_text_len / (CONSUMER_WARPGROUPS * K::qo_height), qo_heads, batch);  // todo5

    if (!process_text) 
    {
        cudaFuncSetAttribute(
            fwd_attend_ker<128, false, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );
        fwd_attend_ker<128, false, false, true><<<grid_image, (32*NUM_WORKERS), mem_size, stream>>>(g, DT, DH, DW, CT, CH, CW);
    }
    else 
    {
        cudaFuncSetAttribute(
            fwd_attend_ker<128, false, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );
        fwd_attend_ker<128, false, true, true><<<grid_text, (32*NUM_WORKERS), mem_size, stream>>>(g, DT, DH, DW, CT, CH, CW);

    }

    return o;
}
