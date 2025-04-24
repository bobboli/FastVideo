from st_attn import sliding_tile_attention
import torch
from einops import rearrange
import math




def tile(x):
    return rearrange(x,
                     "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
                     n_t=5,
                     n_h=6,
                     n_w=10,
                     ts_t=6,
                     ts_h=8,
                     ts_w=8)
def test():
   
   query = torch.ones(1, 115200, 3, 128).to(torch.bfloat16).to("cuda")
   key = torch.ones(1, 115200, 3, 128).to(torch.bfloat16).to("cuda")
   value = torch.ones(1, 115200, 3, 128).to(torch.bfloat16).to("cuda")
   
   encoder_query = torch.ones(1, 540, 3, 128).to(torch.bfloat16).to("cuda")
   encoder_key = torch.ones(1, 540, 3, 128).to(torch.bfloat16).to("cuda")
   encoder_value = torch.ones(1, 540, 3, 128).to(torch.bfloat16).to("cuda")
   
   query = torch.cat([tile(query), encoder_query], dim=1).transpose(1, 2)
   key = torch.cat([tile(key), encoder_key], dim=1).transpose(1, 2)
   value = torch.cat([tile(value), encoder_value], dim=1).transpose(1, 2)
   
   text_length = torch.tensor([361], device='cuda:0', dtype=torch.int32).reshape(-1)
   mask_strategy = [[5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10], [5, 6, 10]]
   
   head_num = query.size(1)
   current_rank = 0
   start_head = current_rank * head_num
   windows = [mask_strategy[head_idx + start_head] for head_idx in range(head_num)]
   
   q_all = query.contiguous()
   k_all = key.contiguous()
   v_all = value.contiguous()
   seq_length = q_all.shape[2]
   
   target_size = math.ceil(seq_length / 384) * 384
   pad_size = target_size - seq_length
   
   if pad_size > 0:
       q_all = torch.cat([q_all, q_all[:, :, -pad_size:]], dim=2)
       k_all = torch.cat([k_all, k_all[:, :, -pad_size:]], dim=2)
       v_all = torch.cat([v_all, v_all[:, :, -pad_size:]], dim=2)
   
   hidden_states = torch.zeros_like(q_all)
   hidden_states = sliding_tile_attention(q_all, k_all, v_all, hidden_states, windows, text_length)
   hidden_states = hidden_states[:, :, :seq_length, :]
   
   
   hidden_states = hidden_states.transpose(1, 2)
   hidden_states = hidden_states.contiguous()

for i in range(100):
    test()
    print(f"pass test {i+1} times")
