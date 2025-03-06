import json
import os

# Define paths as variables
INPUT_MASK_PATH = "/apdcephfs_cq8/share_1367250/mayhmwang/exps/video_speedup/FastVideo/assets/mask_strategy_hunyuan.json"
OUTPUT_MASK_PATH = "/apdcephfs_cq8/share_1367250/mayhmwang/exps/video_speedup/FastVideo/assets/mask_strategy_hunyuan2.json"

# Define scaling factors
SCALE_FACTOR_X = 5
SCALE_FACTOR_Y = 6
SCALE_FACTOR_Z = 10

# Define target dimensions
TARGET_X = 3
TARGET_Y = 7
TARGET_Z = 4




# Load the mask strategy
with open(INPUT_MASK_PATH, 'r') as f:
    mask_strategy = json.load(f)

new_mask_strategy = mask_strategy.copy()

kernel_type0 = set()
kernel_type1 = set()

# Process each entry in the mask strategy
for key, v in new_mask_strategy.items():
    # Calculate scaling ratios
    scaling_ratios = [SCALE_FACTOR_X/v[0], SCALE_FACTOR_Y/v[1], SCALE_FACTOR_Z/v[2]]
    
    # Calculate new dimensions with minimum value of 1
    new_mask_strategy[key] = [
        max(int(TARGET_X/scaling_ratios[0]), 1),
        max(int(TARGET_Y/scaling_ratios[1]), 1),
        max(int(TARGET_Z/scaling_ratios[2]), 1)
    ]
    
    print(f"{v}, {new_mask_strategy[key]}")
    kernel_type0.add(tuple(v))
    kernel_type1.add(tuple(new_mask_strategy[key]))

print(kernel_type0)
print(kernel_type1)

# Uncomment to enable debugging
# import pdb; pdb.set_trace()

# Save the new mask strategy
with open(OUTPUT_MASK_PATH, 'w') as f:
    json.dump(new_mask_strategy, f)