import json
import os

# Define paths as variables
INPUT_MASK_PATH = "assets/mask_strategy_hunyuan.json"

# Define custom kernel size (T, H, W)
CUSTOM_KERNEL_SIZE = (1, 3, 3)  # Time, Height, Width

OUTPUT_MASK_PATH = f"assets/mask_strategy_hunyuan_{CUSTOM_KERNEL_SIZE[0]}_{CUSTOM_KERNEL_SIZE[1]}_{CUSTOM_KERNEL_SIZE[2]}.json"


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