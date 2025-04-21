import math

import torch
from st_attn_cuda import sta_fwd # , sta_fwd_388, sta_fwd_844


def sliding_tile_attention_templatized(q_all, k_all, v_all, img_size, tile_size, window_size, text_length, has_text=True):
    assert has_text, "currently only supports w/ text processing"

    seq_length = q_all.shape[2]
    assert q_all.shape[1] == len(window_size), "Number of heads must match the number of window sizes"

    # TODO: padding text blocks, which should be performed in c++ torch OP 

    hidden_states = torch.empty_like(q_all)
    
    # Process each head with its specific window size parameters
    for batch in range(q_all.shape[0]):
        # First process spatial attention for each head individually
        for head_index, (t_kernel, h_kernel, w_kernel) in enumerate(window_size):
            q_head = q_all[batch:batch + 1, head_index:head_index + 1]
            k_head = k_all[batch:batch + 1, head_index:head_index + 1]
            v_head = v_all[batch:batch + 1, head_index:head_index + 1]
            o_head = hidden_states[batch:batch + 1, head_index:head_index + 1]

            # Call the C++ implementation for spatial attention
            sta_fwd(q_head, k_head, v_head, o_head, torch.tensor(img_size), torch.tensor(tile_size), 
                      torch.tensor([t_kernel, h_kernel, w_kernel]), text_length, False)
        
        # Then process text attention for all heads together if needed
        if has_text:
            q_batch = q_all[batch:batch + 1]
            k_batch = k_all[batch:batch + 1]
            v_batch = v_all[batch:batch + 1]
            o_batch = hidden_states[batch:batch + 1]

            # Call the C++ implementation for text attention
            sta_fwd(q_batch, k_batch, v_batch, o_batch, torch.tensor(img_size), torch.tensor(tile_size),
                      torch.tensor([2, 3, 3]), text_length, True)  # TODO: Does tile_size and kernel_size matter here?
            
    return hidden_states[:, :, :seq_length]



def sliding_tile_attention(q_all, k_all, v_all, window_size, text_length, has_text=True):
    seq_length = q_all.shape[2]
    # print("seq_length====", seq_length)
    if seq_length == 16 * 32 * 56 + 256:
        return sliding_tile_attention_844(q_all, k_all, v_all, window_size, text_length, has_text)  # Dirty fwd


    if has_text:
        assert q_all.shape[
            2] == 115456, "STA currently only supports video with latent size (30, 48, 80), which is 117 frames x 768 x 1280 pixels"
        assert q_all.shape[1] == len(window_size), "Number of heads must match the number of window sizes"
        target_size = math.ceil(seq_length / 384) * 384
        pad_size = target_size - seq_length

        # print(f"Before padding: q_all.shape: {q_all.shape}, q_all.stride: {q_all.stride()}, q_all.is_contiguous: {q_all.is_contiguous()}")
        if pad_size > 0:
            q_all = torch.cat([q_all, q_all[:, :, -pad_size:]], dim=2)
            k_all = torch.cat([k_all, k_all[:, :, -pad_size:]], dim=2)
            v_all = torch.cat([v_all, v_all[:, :, -pad_size:]], dim=2)
        # print(f"After padding: q_all.shape: {q_all.shape}, q_all.stride: {q_all.stride()}, q_all.is_contiguous: {q_all.is_contiguous()}")

    else:
        assert q_all.shape[2] == 82944

    hidden_states = torch.empty_like(q_all)
    # This for loop is ugly. but it is actually quite efficient. The sequence dimension alone can already oversubscribe SMs
    for head_index, (t_kernel, h_kernel, w_kernel) in enumerate(window_size):
        for batch in range(q_all.shape[0]):
            q_head, k_head, v_head, o_head = (q_all[batch:batch + 1, head_index:head_index + 1],
                                              k_all[batch:batch + 1,
                                                    head_index:head_index + 1], v_all[batch:batch + 1,
                                                                                      head_index:head_index + 1],
                                              hidden_states[batch:batch + 1, head_index:head_index + 1])

                        # Debug prints
            # print(f"Q shape: {q_head.shape} stride: {q_head.stride()} contig: {q_head.is_contiguous()}")
            # print(f"K shape: {k_head.shape} stride: {k_head.stride()} contig: {k_head.is_contiguous()}") 
            # print(f"V shape: {v_head.shape} stride: {v_head.stride()} contig: {v_head.is_contiguous()}")
            # print(f"O shape: {o_head.shape} stride: {o_head.stride()} contig: {o_head.is_contiguous()}")

            _ = sta_fwd(q_head, k_head, v_head, o_head, t_kernel, h_kernel, w_kernel, text_length, False, has_text)
    if has_text:
        for batch in range(q_all.shape[0]):
            q_head, k_head, v_head, o_head = (q_all[batch:batch + 1],
                                              k_all[batch:batch + 1], v_all[batch:batch + 1],
                                              hidden_states[batch:batch + 1])   

            _ = sta_fwd(q_head, k_head, v_head, o_head, 3, 3, 3,  text_length, True, True)
    return hidden_states[:, :, :seq_length]


def sliding_tile_attention_388(q_all, k_all, v_all, window_size, text_length, has_text=True):
    seq_length = q_all.shape[2]
    if has_text:
        assert q_all.shape[2] == 18 * 32 * 56 + 256, "STA currently only supports video with latent size (18, 32, 56)"
        assert q_all.shape[1] == len(window_size), "Number of heads must match the number of window sizes"
        target_size = math.ceil(seq_length / 192) * 192
        pad_size = target_size - seq_length
        if pad_size > 0:
            q_all = torch.cat([q_all, q_all[:, :, -pad_size:]], dim=2)
            k_all = torch.cat([k_all, k_all[:, :, -pad_size:]], dim=2)
            v_all = torch.cat([v_all, v_all[:, :, -pad_size:]], dim=2)
    else:
        assert q_all.shape[2] == 82944

    hidden_states = torch.empty_like(q_all)
    # This for loop is ugly. but it is actually quite efficient. The sequence dimension alone can already oversubscribe SMs
    for head_index, (t_kernel, h_kernel, w_kernel) in enumerate(window_size):
        for batch in range(q_all.shape[0]):
            q_head, k_head, v_head, o_head = (q_all[batch:batch + 1, head_index:head_index + 1],
                                              k_all[batch:batch + 1,
                                                    head_index:head_index + 1], v_all[batch:batch + 1,
                                                                                      head_index:head_index + 1],
                                              hidden_states[batch:batch + 1, head_index:head_index + 1])

            _ = sta_fwd_388(q_head, k_head, v_head, o_head, t_kernel, h_kernel, w_kernel, text_length, False, has_text)
    if has_text:
        for batch in range(q_all.shape[0]):
            q_head, k_head, v_head, o_head = (q_all[batch:batch + 1],
                                              k_all[batch:batch + 1], v_all[batch:batch + 1],
                                              hidden_states[batch:batch + 1])   

            _ = sta_fwd_388(q_head, k_head, v_head, o_head, 3, 3, 3,  text_length, True, True)
    return hidden_states[:, :, :seq_length]


def sliding_tile_attention_844(q_all, k_all, v_all, window_size, text_length, has_text=True):
    assert has_text, "currently only supports text processing"
    seq_length = q_all.shape[2]

    assert q_all.shape[2] == 16 * 32 * 56 + 256, "STA currently only supports video with latent size (16, 32, 56) with 256 text tokens"
    assert q_all.shape[1] == len(window_size), "Number of heads must match the number of window sizes"
    assert seq_length % 128 == 0, "sequence length of the image+text tokens must be divisible by 128"


    hidden_states = torch.empty_like(q_all)
    # This for loop is ugly. but it is actually quite efficient. The sequence dimension alone can already oversubscribe SMs
    for head_index, (t_kernel, h_kernel, w_kernel) in enumerate(window_size):
        for batch in range(q_all.shape[0]):
            q_head, k_head, v_head, o_head = (q_all[batch:batch + 1, head_index:head_index + 1],
                                              k_all[batch:batch + 1,
                                                    head_index:head_index + 1], v_all[batch:batch + 1,
                                                                                      head_index:head_index + 1],
                                              hidden_states[batch:batch + 1, head_index:head_index + 1])

            # Debug prints
            # print(f"Q shape: {q_head.shape} stride: {q_head.stride()} contig: {q_head.is_contiguous()}")
            # print(f"K shape: {k_head.shape} stride: {k_head.stride()} contig: {k_head.is_contiguous()}") 
            # print(f"V shape: {v_head.shape} stride: {v_head.stride()} contig: {v_head.is_contiguous()}")
            # print(f"O shape: {o_head.shape} stride: {o_head.stride()} contig: {o_head.is_contiguous()}")

            _ = sta_fwd_844(q_head, k_head, v_head, o_head, t_kernel, h_kernel, w_kernel, text_length, False, has_text)

    for batch in range(q_all.shape[0]):
        q_head, k_head, v_head, o_head = (q_all[batch:batch + 1],
                                            k_all[batch:batch + 1], v_all[batch:batch + 1],
                                            hidden_states[batch:batch + 1])   

        _ = sta_fwd_844(q_head, k_head, v_head, o_head, 2, 3, 3,  text_length, True, has_text)
        
    return hidden_states[:, :, :seq_length]