#!/usr/bin/env python3
import os
import sys
import struct
import math

def read_int(f):
    return struct.unpack('i', f.read(4))[0]

def read_float(f):
    return struct.unpack('f', f.read(4))[0]

def dequantize_bf16(val_u16):
    # BF16 is just the top 16 bits of a 32-bit float
    bits = (val_u16 << 16)
    return struct.unpack('f', struct.pack('I', bits))[0]

def inspect_model(filepath):
    if not os.path.exists(filepath):
        print(f"[-] Error: File not found: {filepath}")
        return

    print(f"[*] Reading model file: {filepath}")
    file_size = os.path.getsize(filepath)
    print(f"[*] File size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")

    with open(filepath, 'rb') as f:
        # 1. Read Header
        magic = read_int(f)
        version = read_int(f)

        if magic != 0x47505432:
            print(f"[-] Error: Invalid magic number: 0x{magic:X}. Expected 0x47505432.")
            return

        vocab_size = read_int(f)
        max_seq_len = read_int(f)
        embedding_dim = read_int(f)
        num_heads = read_int(f)
        num_layers = read_int(f)

        tokenizer_type = 0 # 0=CHAR default
        quantization_level = 0 # 0=FP32 default

        if version == 1:
            print("[+] Version 1 (Legacy CHAR/FP32 model)")
        elif version == 2:
            tokenizer_type = read_int(f)
            quantization_level = read_int(f)
            print(f"[+] Version 2 Model")
        else:
            print(f"[-] Error: Unsupported version {version}")
            return

        print(f"    Vocab Size:         {vocab_size}")
        print(f"    Max Seq Length:     {max_seq_len}")
        print(f"    Embedding Dim:      {embedding_dim}")
        print(f"    Num Heads:          {num_heads}")
        print(f"    Num Layers:         {num_layers}")
        print(f"    Tokenizer Type:     {tokenizer_type} ({'CHAR' if tokenizer_type == 0 else 'BPE'})")
        print(f"    Quantization Level: {quantization_level} ({'FP32' if quantization_level == 0 else ('BF16' if quantization_level == 1 else 'INT8')})")
        print("-" * 60)

        # 2. Read Vocabulary
        if tokenizer_type == 0:
            vocab_chars = f.read(vocab_size)
            print(f"[+] Read character vocabulary: {repr(vocab_chars[:50])}...")
            
            # Align padding to 4-byte boundaries
            vocab_bytes = vocab_size * 1
            padding = (4 - (vocab_bytes % 4)) % 4
            if padding > 0:
                f.read(padding)
        else:
            num_merges = read_int(f)
            print(f"[+] Read BPE token merges rule size: {num_merges}")
            merges = []
            for _ in range(num_merges):
                left = read_int(f)
                right = read_int(f)
                merges.append((left, right))
            print(f"    First 5 merges: {merges[:5]}")
            print(f"    Last 5 merges:  {merges[-5:]}")
        
        print("-" * 60)

        # Define parameter shapes/names sequentially
        # Order must match get_parameters() in C++
        params_meta = []
        
        # wte
        params_meta.append(("wte", (vocab_size, embedding_dim)))
        # wpe
        params_meta.append(("wpe", (max_seq_len, embedding_dim)))
        
        # blocks
        for l in range(num_layers):
            params_meta.append((f"blocks.{l}.ln1_gamma", (embedding_dim,)))
            params_meta.append((f"blocks.{l}.ln1_beta", (embedding_dim,)))
            params_meta.append((f"blocks.{l}.w_qkv", (embedding_dim, 3 * embedding_dim)))
            params_meta.append((f"blocks.{l}.b_qkv", (3 * embedding_dim,)))
            params_meta.append((f"blocks.{l}.w_proj", (embedding_dim, embedding_dim)))
            params_meta.append((f"blocks.{l}.b_proj", (embedding_dim,)))
            params_meta.append((f"blocks.{l}.ln2_gamma", (embedding_dim,)))
            params_meta.append((f"blocks.{l}.ln2_beta", (embedding_dim,)))
            params_meta.append((f"blocks.{l}.w_fc", (embedding_dim, 4 * embedding_dim)))
            params_meta.append((f"blocks.{l}.b_fc", (4 * embedding_dim,)))
            params_meta.append((f"blocks.{l}.w_proj_mlp", (4 * embedding_dim, embedding_dim)))
            params_meta.append((f"blocks.{l}.b_proj_mlp", (embedding_dim,)))
            
        # ln_f
        params_meta.append(("ln_f_gamma", (embedding_dim,)))
        params_meta.append(("ln_f_beta", (embedding_dim,)))

        total_weights_count = 0
        wte_weights = []
        
        print(f"{'Parameter Name':<30} | {'Shape':<15} | {'Scale Factor':<12} | {'Min':<8} | {'Max':<8} | {'Mean':<8}")
        print("=" * 95)
        
        for name, shape in params_meta:
            num_elements = 1
            for dim in shape:
                num_elements *= dim
            total_weights_count += num_elements

            # Read parameter according to quantization level
            scale = 1.0
            raw_weights = []

            if quantization_level == 0:
                # FP32
                raw_bytes = f.read(num_elements * 4)
                raw_weights = list(struct.unpack(f'{num_elements}f', raw_bytes))
            elif quantization_level == 1:
                # BF16
                raw_bytes = f.read(num_elements * 2)
                bf16_ints = list(struct.unpack(f'{num_elements}H', raw_bytes))
                raw_weights = [dequantize_bf16(x) for x in bf16_ints]
                
                # Check for 2-byte alignment padding
                bytes_read = num_elements * 2
                if bytes_read % 4 != 0:
                    f.read(2) # skip 2 padding bytes
            elif quantization_level == 2:
                # INT8
                scale = read_float(f)
                raw_bytes = f.read(num_elements)
                int8_vals = list(struct.unpack(f'{num_elements}b', raw_bytes))
                raw_weights = [float(x) * scale for x in int8_vals]
                
                # Check for alignment padding (align to 4-byte boundaries)
                bytes_written = 4 + num_elements # scale (4 bytes) + int8 array
                padding = (4 - (bytes_written % 4)) % 4
                if padding > 0:
                    f.read(padding) # skip padding bytes

            if name == "wte":
                wte_weights = raw_weights

            # Calculate stats
            w_min = min(raw_weights)
            w_max = max(raw_weights)
            w_mean = sum(raw_weights) / len(raw_weights)
            
            shape_str = "x".join(map(str, shape))
            scale_str = f"{scale:.6e}" if quantization_level == 2 else "N/A"
            print(f"{name:<30} | {shape_str:<15} | {scale_str:<12} | {w_min:<8.4f} | {w_max:<8.4f} | {w_mean:<8.4f}")

        print("=" * 95)
        print(f"[+] Total Parameters count: {total_weights_count:,}")
        
        # Verify if we reached EOF
        remaining = len(f.read())
        if remaining > 0:
            print(f"[!] Warning: There are {remaining} trailing bytes left unread in the file.")
        else:
            print("[+] Successfully parsed the entire binary file with 100% boundary check.")

        # Visceral numerical proof: Output a small patch of floats!
        if len(wte_weights) > 0:
            print("\n" + "=" * 95)
            print("🔢 VISCERAL NUMERICAL PROOF: A 5x5 Matrix Slice of Word Token Embeddings (wte)")
            print("=" * 95)
            print("Here are the actual float values stored inside the first 5 rows (tokens) and first 5 columns (dims):")
            print("Row Index (Token) | Col 0     Col 1     Col 2     Col 3     Col 4")
            print("-" * 75)
            for token_id in range(5):
                slice_vals = wte_weights[token_id * embedding_dim : token_id * embedding_dim + 5]
                slice_str = "  ".join(f"{x:+9.6f}" for x in slice_vals)
                print(f"  Token ID {token_id:03d}     | {slice_str}")
            print("-" * 75)
            print("Here is a raw list of the first 1000 float numbers in the model:")
            print(", ".join(f"{x:.6f}" for x in wte_weights[:1000]))
            print("=" * 95)

if __name__ == "__main__":
    filepath = "dataset/macbeth2.bin"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    inspect_model(filepath)
