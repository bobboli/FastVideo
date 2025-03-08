import torch
import triton.testing
from st_attn import sliding_tile_attention_844
import numpy as np
import matplotlib.pyplot as plt

# Parameters for benchmarking
B = 2  # batch size
H = 24  # number of heads
D = 128  # head dimension
dtype = torch.bfloat16
device = "cuda"

# Text processing parameters
text_length = 57
text_max_len = 256

# Video dimensions
tile_size = (8, 4, 4)
latent_size = (16, 32, 56)
N = latent_size[0] * latent_size[1] * latent_size[2] + text_max_len  # total sequence length

def calculate_flop(batch, seqlen, nheads, headdim, tile_seqlen=None, text_len=0):
    """
    Calculate FLOP (Floating Point Operations) for different attention mechanisms
    
    For standard attention: 4 * batch * seqlen^2 * nheads * headdim
    For sliding tile: 4 * batch * seqlen * tile_seqlen * nheads * headdim
    """
    video_len = seqlen - text_max_len
    
    if tile_seqlen is None:  # Flash
        # Standard attention (full quadratic complexity)
        # Text-to-text: text_len^2
        # Video-to-video: video_len^2 
        # Text-to-video & Video-to-text: 2 * text_len * video_len
        flop = 4 * batch * nheads * headdim * (
            text_len**2 + video_len**2 + 2 * text_len * video_len
        )

        print(f"Flash: seq_len {N}, text_len {text_len}, video_len {video_len}, flop {flop}")
    else:
        # Sliding tile attention (reduced complexity for video tokens)
        # Text-to-text: text_len^2 (unchanged)
        # Video-to-video: video_len * tile_seqlen (windowed)
        # Text-to-video & Video-to-text: 2 * text_len * video_len (unchanged)
        flop = 4 * batch * nheads * headdim * (
            text_len**2 + video_len * tile_seqlen + 2 * text_len * video_len
        )
        print(f"STA: seq_len {N}, text_len {text_len}, video_len {video_len}, tile_seqlen {tile_seqlen}, flop {flop}")
    
    return flop

def main():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Create random tensors
    q = torch.randn((B, H, N, D), dtype=dtype, device=device).contiguous()
    k = torch.randn((B, H, N, D), dtype=dtype, device=device).contiguous()
    v = torch.randn((B, H, N, D), dtype=dtype, device=device).contiguous()
    
    # Only use supported kernel sizes from the implementation
    kernel_sizes = [
        [1, 1, 1],
        [1, 3, 3],
        [1, 3, 7],
        [1, 5, 5],
        [1, 5, 7],
        [1, 5, 11],
        [1, 8, 11],
        [1, 8, 14],
        [2, 1, 1],
        [2, 3, 3],
        [2, 3, 7],
        [2, 5, 5],
        [2, 5, 7],
        [2, 5, 11],
        [2, 8, 11],
        [2, 8, 14]
    ]
    
    # Store results
    results = {}
    flash_time_ms = None
    
    # Reference FLOP for full attention
    full_attention_flop = calculate_flop(B, N, H, D, None, text_length)
    print(f"Full attention theoretical FLOP: {full_attention_flop/1e12:.2f} TFLOP")
    
    # Try to benchmark Flash Attention first (for baseline)
    try:
        from flash_attn import flash_attn_func
        
        print("\nBenchmarking Flash Attention (baseline):")
        q_flash = q.permute(0, 2, 1, 3)  # [B, N, H, D]
        k_flash = k.permute(0, 2, 1, 3)
        v_flash = v.permute(0, 2, 1, 3)
        
        def benchmark_flash():
            return flash_attn_func(q_flash, k_flash, v_flash) 
        
        flash_time_ms = triton.testing.do_bench(benchmark_flash, warmup=10, rep=100)
        throughput = full_attention_flop / (flash_time_ms / 1000)
        
        results["Flash Attention"] = {
            'time_ms': flash_time_ms,
            'tile_seqlen': N,  # Full attention tile
            'theoretical_flop': full_attention_flop,
            'tflops': throughput / 1e12,
            'theoretical_speedup': 1.0,  # Reference speedup
            'actual_speedup': 1.0   # Reference speedup
        }
        
        print(f"  Average time: {flash_time_ms:.3f} ms")
        print(f"  Throughput: {throughput/1e12:.2f} TFLOP/s")
    except ImportError:
        print("Flash Attention not available")
    
    # Benchmark each kernel size
    print(f"\nBenchmarking STA-844 with Text Processing (seq_len={N}, text_len={text_length}):")
    for ks in kernel_sizes:
        ks_str = f"[{ks[0]}, {ks[1]}, {ks[2]}]"
        tile_seqlen = ks[0] * ks[1] * ks[2] * tile_size[0] * tile_size[1] * tile_size[2]
        sta_flop = calculate_flop(B, N, H, D, tile_seqlen, text_length)
        
        # Calculate theoretical speedup compared to full attention
        theoretical_speedup = full_attention_flop / sta_flop
        
        print(f"\nTesting kernel size: {ks_str} (tile_seqlen: {tile_seqlen})")
        print(f"  Theoretical FLOP: {sta_flop/1e12:.2f} TFLOP")
        print(f"  Theoretical speedup: {theoretical_speedup:.2f}x")
        
        # Define function to benchmark (with text processing enabled)
        def benchmark_fn():
            return sliding_tile_attention_844(q, k, v, [ks] * H, text_length, has_text=True)
        
        # Benchmark using triton's utility
        try:
            ms = triton.testing.do_bench(benchmark_fn, warmup=10, rep=50)
            throughput = sta_flop / (ms / 1000)  # FLOP/s
            
            # Calculate actual speedup compared to Flash Attention if available
            actual_speedup = flash_time_ms / ms if flash_time_ms is not None else float('nan')
            
            results[ks_str] = {
                'time_ms': ms,
                'tile_seqlen': tile_seqlen,
                'theoretical_flop': sta_flop,
                'tflops': throughput / 1e12,
                'theoretical_speedup': theoretical_speedup,
                'actual_speedup': actual_speedup
            }
            
            print(f"  Average time: {ms:.3f} ms")
            print(f"  Throughput: {throughput/1e12:.2f} TFLOP/s")
            if flash_time_ms is not None:
                print(f"  Actual speedup vs Flash: {actual_speedup:.2f}x")
        except Exception as e:
            print(f"  Error with kernel size {ks_str}: {str(e)}")
            results[ks_str] = {
                'time_ms': float('nan'),
                'tile_seqlen': tile_seqlen,
                'theoretical_flop': sta_flop,
                'tflops': float('nan'),
                'theoretical_speedup': theoretical_speedup,
                'actual_speedup': float('nan')
            }
    
    # Filter out failed benchmarks and sort by execution time
    valid_results = {k: v for k, v in results.items() if not np.isnan(v['time_ms'])}
    sorted_results = dict(sorted(valid_results.items(), key=lambda item: item[1]['time_ms']))
    
    if sorted_results:
        # Create visualization with multiple plots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Execution Time
        plt.subplot(2, 2, 1)
        kernel_labels = list(sorted_results.keys())
        times = [v['time_ms'] for v in sorted_results.values()]
        
        bars = plt.bar(kernel_labels, times, color='skyblue')
        plt.ylabel('Execution Time (ms)', fontsize=12)
        plt.title('Execution Time by Kernel Size', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', fontsize=8)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 2: Throughput (TFLOP/s)
        plt.subplot(2, 2, 2)
        throughputs = [v['tflops'] for v in sorted_results.values()]
        
        bars = plt.bar(kernel_labels, throughputs, color='lightgreen')
        plt.ylabel('Throughput (TFLOP/s)', fontsize=12)
        plt.title('Computational Throughput', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', fontsize=8)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot 3: Tile Sequence Length vs Time
        plt.subplot(2, 2, 3)
        tile_seqlens = [v['tile_seqlen'] for v in sorted_results.values()]
        
        # Create scatter plot with connected lines
        plt.plot(tile_seqlens, times, 'o-', color='coral')
        plt.xscale('log')  # Log scale for tile_seqlen
        plt.xlabel('Tile Sequence Length (log scale)', fontsize=12)
        plt.ylabel('Execution Time (ms)', fontsize=12)
        plt.title('Execution Time vs Tile Sequence Length', fontsize=14)
        plt.grid(True, alpha=0.7)
        
        # Annotate points with kernel size
        for i, txt in enumerate(kernel_labels):
            plt.annotate(txt, (tile_seqlens[i], times[i]), fontsize=7, 
                        xytext=(5, 5), textcoords='offset points')
        
        # Plot 4: Theoretical vs Actual Speedup
        plt.subplot(2, 2, 4)
        
        # Only include points where we have actual speedup data
        valid_indices = [i for i, k in enumerate(kernel_labels) 
                        if not np.isnan(sorted_results[k]['actual_speedup'])]
        
        # Skip this plot if we don't have Flash Attention data
        if valid_indices and 'Flash Attention' in sorted_results:
            theoretical_speedups = [sorted_results[kernel_labels[i]]['theoretical_speedup'] for i in valid_indices]
            actual_speedups = [sorted_results[kernel_labels[i]]['actual_speedup'] for i in valid_indices]
            plot_labels = [kernel_labels[i] for i in valid_indices]
            
            # Create scatter plot
            plt.scatter(theoretical_speedups, actual_speedups, 
                      c=[sorted_results[kernel_labels[i]]['tile_seqlen'] for i in valid_indices], 
                      cmap='viridis', s=100, alpha=0.7)
            
            # Add diagonal line representing perfect correlation
            max_val = max(max(theoretical_speedups), max(actual_speedups)) * 1.1
            plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
            
            plt.colorbar(label='Tile Sequence Length')
            plt.xlabel('Theoretical Speedup vs Full Attention', fontsize=12)
            plt.ylabel('Actual Speedup vs Flash Attention', fontsize=12)
            plt.title('Theoretical vs Actual Speedup', fontsize=14)
            plt.grid(True, alpha=0.7)
            
            # Annotate points with kernel size
            for i, txt in enumerate(plot_labels):
                plt.annotate(txt, (theoretical_speedups[i], actual_speedups[i]), 
                           fontsize=7, xytext=(5, 5), textcoords='offset points')
        else:
            plt.text(0.5, 0.5, 'Flash Attention not available for comparison', 
                   ha='center', va='center', fontsize=12)
            plt.axis('off')
        
        plt.suptitle(f'STA-844 Performance with Text Processing\n(seq_len={N}, text_len={text_length})', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('sta_844_benchmark.png', dpi=300)
        print("\nPlot saved as sta_844_benchmark.png")
    else:
        print("\nNo valid results to plot")
    
    # Print summary table sorted by execution time
    print("\nSummary Table (sorted by execution time):")
    print(f"{'Kernel Size':<15} {'Time (ms)':<10} {'TFLOP/s':<10} {'Tile SeqLen':<12} {'Theoritical Speedup':<15} {'Actual Speedup':<15}")
    print("-" * 80)
    
    for k, v in sorted_results.items():
        actual_speedup = v['actual_speedup']
        actual_speedup_str = f"{actual_speedup:.2f}" if not np.isnan(actual_speedup) else "N/A"
        
        print(f"{k:<15} {v['time_ms']:<10.3f} {v['tflops']:<10.2f} {v['tile_seqlen']:<12} {v['theoretical_speedup']:<15.2f} {actual_speedup_str:<15}")
    
    # Print failed kernel sizes
    failed = [k for k, v in results.items() if np.isnan(v['time_ms'])]
    if failed:
        print("\nFailed kernel sizes:")
        for k in failed:
            print(f"  {k}")
    
    return results

if __name__ == "__main__":
    main() 