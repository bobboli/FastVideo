import torch
from flex_sta_ref import get_sliding_tile_attention_mask
from st_attn import sliding_tile_attention
from torch.nn.attention.flex_attention import flex_attention
from tqdm import tqdm

print("Testing 388 implementation for 18x32x56 spatial dimensions")

flex_attention = torch.compile(flex_attention, dynamic=False)

def flex_test(Q, K, V, kernel_size):
    # For 18x32x56 = 32,256 (often with seq_len 82944)
    mask = get_sliding_tile_attention_mask(kernel_size, (3, 4, 4), (18, 32, 56), 256, 'cuda', 0)
    output = flex_attention(Q, K, V, block_mask=mask)
    return output

def h100_388_fwd_kernel_test(Q, K, V, kernel_size):
    # Will dispatch to the 388 implementation because seq_len = 82944
    o = sliding_tile_attention(Q, K, V, [kernel_size] * Q.shape[1], 256, False)
    return o

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    return scaled_tensor.contiguous()

def check_correctness(b, h, n, d, causal, mean, std, num_iterations=50, error_mode='all'):
    results = {
        'H100_388 vs FLEX': {
            'sum_diff': 0,
            'sum_abs': 0,
            'max_diff': 0
        },
    }
    
    # Appropriate kernel sizes for 388 implementation
    kernel_size_ls = [
        (3, 3, 3),  # 0x111
        (3, 3, 5),  # 0x112
        (5, 3, 3),  # 0x211
        (3, 5, 5),  # 0x122
        (5, 6, 1),  # 0x230
        (5, 3, 5),  # 0x212
        (5, 5, 5),  # 0x222
        (5, 5, 7),  # 0x223
    ]
    
    from tqdm import tqdm
    for kernel_size in tqdm(kernel_size_ls):
        for _ in range(num_iterations):
            torch.manual_seed(0)

            Q = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            K = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            V = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            
            h100_o = h100_388_fwd_kernel_test(Q, K, V, kernel_size)
            flex_o = flex_test(Q, K, V, kernel_size)

            diff = flex_o - h100_o
            abs_diff = torch.abs(diff)
            results['H100_388 vs FLEX']['sum_diff'] += torch.sum(abs_diff).item()
            results['H100_388 vs FLEX']['max_diff'] = max(
                results['H100_388 vs FLEX']['max_diff'], 
                torch.max(abs_diff).item()
            )

            torch.cuda.empty_cache()
            
        print("kernel_size", kernel_size)
        print("max_diff", torch.max(abs_diff).item())
        print(
            "avg_diff",
            torch.sum(abs_diff).item() / (b * h * n * d *
                                         (1 if error_mode == 'output' else 3 if error_mode == 'backward' else 4)))

    total_elements = b * h * n * d * num_iterations * (1 if error_mode == 'output' else
                                                     3 if error_mode == 'backward' else 4) * len(kernel_size_ls)
    for name, data in results.items():
        avg_diff = data['sum_diff'] / total_elements
        max_diff = data['max_diff']
        results[name] = {'avg_diff': avg_diff, 'max_diff': max_diff}

    return results

def generate_error_graphs(b, h, d, causal, mean, std, error_mode='all'):
    # Use exactly 82944 for sequence length to test 388 implementation
    seq_lengths = [82944]  # 18x32x56 dimensions

    h100_avg_errors, h100_max_errors = [], []

    for n in tqdm(seq_lengths, desc="Generating error data"):
        results = check_correctness(b, h, n, d, causal, mean, std, error_mode=error_mode)

        h100_avg_errors.append(results['H100_388 vs FLEX']['avg_diff'])
        h100_max_errors.append(results['H100_388 vs FLEX']['max_diff'])
        
    print(f"Average error: {h100_avg_errors[0]}")
    print(f"Maximum error: {h100_max_errors[0]}")

# Example usage
b, h, d = 2, 12, 128
causal = False
mean = 1e-1
std = 10

print("Testing H100 388 implementation")
for mode in ['output']:
    print(f"Testing mode: {mode}")
    generate_error_graphs(b, h, d, causal, mean, std, error_mode=mode)

print("Testing completed for H100 388 implementation.")

