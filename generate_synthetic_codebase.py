#!/usr/bin/env python3
import os
import random
import string

def generate_random_function(name):
    # Generates a dummy function
    return f"def {name}():\n    pass\n\n"

def main():
    os.makedirs("inputs", exist_ok=True)
    depths = [5000, 15000, 30000] # Tokens approx (words)
    needles = [
        "BUG: Hardcoded secret key is 'homelab_sre_secret_xyz'",
        "TODO: The API gateway routing logic is broken here",
        "CRITICAL: Race condition in SQLite async worker detected here"
    ]
    
    for i, (depth, needle) in enumerate(zip(depths, needles)):
        filename = f"inputs/synthetic_codebase_{depth}.py"
        with open(filename, "w") as f:
            for j in range(depth // 10): # Approx 10 tokens per function
                f.write(generate_random_function(f"dummy_func_{j}"))
                if j == (depth // 20): # Insert needle around the middle of the file
                    f.write(f"# {needle}\n")
        print(f"Generated {filename} with needle '{needle}'")

if __name__ == "__main__":
    main()
