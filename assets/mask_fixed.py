import json
import os

# Define paths as variables
INPUT_MASK_PATH = "mask_strategy_hunyuan.json"
OUTPUT_MASK_PATH = "mask_strategy_hunyuan_2_8_14.json"

# Define custom kernel size (T, H, W)
CUSTOM_KERNEL_SIZE = (2, 8, 14)  # Time, Height, Width

# Load the mask strategy
with open(INPUT_MASK_PATH, 'r') as f:
    mask_strategy = json.load(f)

new_mask_strategy = mask_strategy.copy()

# Process each entry in the mask strategy
for key, v in new_mask_strategy.items():
    # Store original values
    original_values = v.copy()
    
    # Replace with custom kernel size
    new_mask_strategy[key] = list(CUSTOM_KERNEL_SIZE)
    
    print(f"{original_values}, {new_mask_strategy[key]}")

# Uncomment to enable debugging
# import pdb; pdb.set_trace()

# Save the new mask strategy
with open(OUTPUT_MASK_PATH, 'w') as f:
    json.dump(new_mask_strategy, f)