import torch
from flex_sta_ref import get_sliding_tile_attention_mask
from st_attn import sliding_tile_attention_388
from torch.nn.attention.flex_attention import flex_attention
from tqdm import tqdm

print("Testing 388 implementation with text processing enabled")

flex_attention = torch.compile(flex_attention, dynamic=False)

num_heads = 24


# Example usage
b, h, d = 2, 24, 128
causal = False
mean = 1e-1
std = 10

def flex_test(Q, K, V, kernel_size):
    # Text-specific parameters for 388 implementation
    mask = get_sliding_tile_attention_mask(kernel_size, (3, 8, 8), (18, 32, 56), 256, 'cuda', 256)
    print(Q.shape, K.shape, V.shape, mask.shape)
    output = flex_attention(Q, K, V, block_mask=mask)
    return output


def h100_fwd_kernel_test(Q, K, V, kernel_size):
    # Enable text processing with has_text=True
    o = sliding_tile_attention_388(Q, K, V, [kernel_size] * h, 256, has_text=True)
    return o


def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    return scaled_tensor.contiguous()


def check_correctness(b, h, n, d, causal, mean, std, num_iterations=10, error_mode='all'):
    results = {
        'TK vs FLEX': {
            'sum_diff': 0,
            'sum_abs': 0,
            'max_diff': 0
        },
    }
    
    # Kernel sizes appropriate for text processing
    kernel_size_ls = [(3, 3, 3), (5, 5, 7), (3, 5, 5)]
    
    from tqdm import tqdm
    for kernel_size in tqdm(kernel_size_ls):
        print(f"\nTesting kernel size: {kernel_size}")
        for iter_num in range(num_iterations):
            print(f"  Iteration {iter_num+1}/{num_iterations}")
            torch.manual_seed(iter_num)  # Different seed for better coverage

            Q = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            K = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            V = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
            
            try:
                tk_o = h100_fwd_kernel_test(Q, K, V, kernel_size)
                pt_o = flex_test(Q, K, V, kernel_size)

                diff = pt_o - tk_o
                abs_diff = torch.abs(diff)
                current_max_diff = torch.max(abs_diff).item()
                current_avg_diff = torch.sum(abs_diff).item() / (b * h * n * d)
                
                results['TK vs FLEX']['sum_diff'] += torch.sum(abs_diff).item()
                results['TK vs FLEX']['max_diff'] = max(
                    results['TK vs FLEX']['max_diff'], 
                    current_max_diff
                )
                
                print(f"    Max diff: {current_max_diff:.6e}")
                print(f"    Avg diff: {current_avg_diff:.6e}")
            
            except Exception as e:
                print(f"    Error with kernel size {kernel_size}, iteration {iter_num}: {str(e)}")
            
            # Clear cache after each iteration
            torch.cuda.empty_cache()
            
        print(f"\nKernel size {kernel_size} summary:")
        print(f"  Max diff: {results['TK vs FLEX']['max_diff']:.6e}")

    total_elements = b * h * n * d * num_iterations * len(kernel_size_ls)
    for name, data in results.items():
        avg_diff = data['sum_diff'] / total_elements
        max_diff = data['max_diff']
        results[name] = {'avg_diff': avg_diff, 'max_diff': max_diff}
    
    print("\nOverall results:")
    print(f"Average difference: {results['TK vs FLEX']['avg_diff']:.6e}")
    print(f"Maximum difference: {results['TK vs FLEX']['max_diff']:.6e}")

    return results


def generate_error_graphs(b, h, d, causal, mean, std, error_mode='all'):
    # Use the sequence length for 388 implementation
    seq_lengths = [18 * 32 * 56 + 256]

    tk_avg_errors, tk_max_errors = [], []

    for n in tqdm(seq_lengths, desc="Testing with text processing"):
        results = check_correctness(b, h, n, d, causal, mean, std, error_mode=error_mode)

        tk_avg_errors.append(results['TK vs FLEX']['avg_diff'])
        tk_max_errors.append(results['TK vs FLEX']['max_diff'])
        
    print(f"Average error: {tk_avg_errors[0]:.6e}")
    print(f"Maximum error: {tk_max_errors[0]:.6e}")


print("=" * 50)
print("TESTING 388 IMPLEMENTATION WITH TEXT PROCESSING")
print("=" * 50)

for mode in ['output']:
    print(f"Testing mode: {mode}")
    generate_error_graphs(b, h, d, causal, mean, std, error_mode=mode)

print("Text processing tests completed for 388 implementation.") 