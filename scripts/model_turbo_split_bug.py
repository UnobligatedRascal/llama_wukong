#!/usr/bin/env python3
"""
Model the turbo2_0/turbo3_0 KV cache split bug in ggml-backend-meta.cpp.

The meta backend uses linear stride scaling for split tensors:
    nb[i] = tensor->nb[i] * ne[split_dim]/tensor->ne[split_dim]

For quantized types, nb[1] must be block-aligned:
    nb[1] = ggml_row_size(type, ne[0]) = (ne[0]/blck_size) * sizeof(block)

Linear scaling produces incorrect strides for turbo types.
"""

import math

# Turbo type parameters
TURBO_TYPES = {
    "turbo4_0": {"blck_size": 128, "block_bytes": 68},    # 4.25 bpw
    "turbo3_0": {"blck_size": 128, "block_bytes": 50},    # 3.125 bpw  
    "turbo2_0": {"blck_size": 128, "block_bytes": 34},    # 2.125 bpw
}

# Standard types for comparison
STD_TYPES = {
    "q4_0": {"blck_size": 32, "block_bytes": 17},
    "q8_0": {"blck_size": 32, "block_bytes": 32},
}

def row_size(ne0, blck_size, block_bytes):
    """Compute actual row size for quantized type."""
    return math.ceil(ne0 / blck_size) * block_bytes

def linear_scaled_stride(full_nb1, full_ne0, split_ne0):
    """Meta backend's linear stride scaling (integer division)."""
    return full_nb1 * split_ne0 // full_ne0

def simulate_split(tensor_ne0, n_gpus, type_name, type_params):
    """Simulate splitting a KV cache tensor across GPUs."""
    blck_size = type_params["blck_size"]
    block_bytes = type_params["block_bytes"]
    
    full_nb1 = row_size(tensor_ne0, blck_size, block_bytes)
    chunk_size_full = full_nb1  # meta backend uses this
    
    print(f"\n{'='*70}")
    print(f"Type: {type_name}, ne[0]={tensor_ne0}, GPUs={n_gpus}")
    print(f"  Block size={blck_size}, Block bytes={block_bytes}")
    print(f"  Full tensor nb[1]={full_nb1}, chunk_size_full={chunk_size_full}")
    
    # Compute split sizes (simplified: equal split, aligned to block_size)
    base_split = tensor_ne0 // n_gpus
    remainder = tensor_ne0 % n_gpus
    
    splits = []
    for gpu in range(n_gpus):
        split_size = base_split + (1 if gpu < remainder else 0)
        # Align down to block_size (as meta backend does)
        split_size_aligned = (split_size // blck_size) * blck_size
        if split_size_aligned == 0 and split_size > 0:
            split_size_aligned = blck_size  # minimum one block
        splits.append(split_size_aligned)
    
    # Adjust last GPU to account for total
    total = sum(splits)
    if total != tensor_ne0:
        # This can happen with alignment - last GPU gets remainder
        splits[-1] = tensor_ne0 - sum(splits[:-1])
    
    print(f"\n  Split sizes (ne[0] per GPU): {splits}")
    
    # Check each GPU's stride
    for gpu, split_ne0 in enumerate(splits):
        actual_nb1 = row_size(split_ne0, blck_size, block_bytes)
        scaled_nb1 = linear_scaled_stride(full_nb1, tensor_ne0, split_ne0)
        
        mismatch = actual_nb1 != scaled_nb1
        
        # Simulate accessing all rows: size = ne[1] * nb[1]
        # For meta backend assertion: size % chunk_size_full == 0
        # If we access the full split tensor:
        #   size_from_scaled = split_ne0_elements * scaled_nb1_per_element_row
        # But the actual data layout uses actual_nb1
        
        # The assertion failure scenario:
        # When meta backend computes size based on scaled strides,
        # it may not be divisible by chunk_size_full
        
        # Example: accessing N rows through meta backend
        # Meta uses scaled_nb1 for offsets, but actual data has actual_nb1
        # This causes size calculations to be wrong
        
        status = "MISMATCH!" if mismatch else "OK"
        print(f"  GPU{gpu}: ne[0]={split_ne0:5d}, "
              f"actual_nb1={actual_nb1:5d}, "
              f"scaled_nb1={scaled_nb1:5d}, "
              f"{status}")
        
        if mismatch:
            # Show the assertion failure scenario
            # If meta backend thinks row stride is scaled_nb1 but actual is actual_nb1
            # Then accessing split_ne0/blck_size blocks gives:
            blocks = split_ne0 // blck_size
            size_via_scaled = blocks * scaled_nb1
            size_via_actual = blocks * actual_nb1
            
            div_scaled = size_via_scaled % chunk_size_full == 0
            div_actual = size_via_actual % chunk_size_full == 0
            
            print(f"    Blocks={blocks}, size_via_scaled={size_via_scaled}, "
                  f"size_via_actual={size_via_actual}")
            print(f"    size_via_scaled % chunk_size_full == 0: {div_scaled}")
            print(f"    size_via_actual % chunk_size_full == 0: {div_actual}")
    
    return splits

def test_scenarios():
    """Test various KV cache configurations."""
    
    # Realistic KV cache dimensions for Qwen3.6-27B
    # n_embd_k_gqa = 4096 (for 27B model with GQA)
    # Split across 4 or 8 GPUs
    
    print("Testing turbo types with realistic KV cache dimensions:")
    
    for type_name, params in TURBO_TYPES.items():
        simulate_split(4096, 4, type_name, params)
        simulate_split(4096, 8, type_name, params)
    
    # Edge case: dimension not evenly divisible by block_size * n_gpus
    print("\n\nEdge case: ne[0] not evenly divisible:")
    for type_name, params in TURBO_TYPES.items():
        simulate_split(4050, 8, type_name, params)
    
    # Compare with standard types that work
    print("\n\nStandard types (for comparison):")
    for type_name, params in STD_TYPES.items():
        simulate_split(4096, 8, type_name, params)

def demonstrate_fix():
    """Show how the fix would work."""
    print("\n\n" + "="*70)
    print("DEMONSTRATING THE FIX")
    print("="*70)
    
    # Use turbo2_0 with problematic split
    type_name = "turbo2_0"
    params = TURBO_TYPES[type_name]
    blck_size = params["blck_size"]
    block_bytes = params["block_bytes"]
    
    full_ne0 = 4050
    split_ne0 = 448  # First GPU's aligned split
    
    full_nb1 = row_size(full_ne0, blck_size, block_bytes)
    
    # BUGGY: linear scaling
    buggy_nb1 = full_nb1 * split_ne0 // full_ne0
    
    # FIXED: use actual row size
    fixed_nb1 = row_size(split_ne0, blck_size, block_bytes)
    
    print(f"\n{type_name}: full_ne0={full_ne0}, split_ne0={split_ne0}")
    print(f"  Full tensor nb[1] = {full_nb1}")
    print(f"  BUGGY (linear scaled) nb[1] = {buggy_nb1}")
    print(f"  FIXED (actual row_size) nb[1] = {fixed_nb1}")
    print(f"  Difference: {abs(fixed_nb1 - buggy_nb1)} bytes")
    
    # Show assertion behavior
    blocks = split_ne0 // blck_size
    chunk_size_full = full_nb1
    
    size_buggy = blocks * buggy_nb1
    size_fixed = blocks * fixed_nb1
    
    print(f"\n  Accessing {blocks} blocks:")
    print(f"  BUGGY: size={size_buggy}, size % chunk_size_full({chunk_size_full}) = {size_buggy % chunk_size_full}")
    print(f"  FIXED: size={size_fixed}, size % chunk_size_full({chunk_size_full}) = {size_fixed % chunk_size_full}")

if __name__ == "__main__":
    test_scenarios()
    demonstrate_fix()
