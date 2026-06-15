import os
import glob

def generate_realistic_haystack(repo_path, output_file, num_chars, needle, needle_pos_ratio):
    # Find all source files
    files = []
    for ext in ['*.md', '*.py', '*.go', '*.yaml']:
        files.extend(glob.glob(os.path.join(repo_path, '**', ext), recursive=True))
    
    haystack = ""
    for f in files:
        try:
            with open(f, 'r') as file:
                haystack += f"\\n\\n--- FILE: {f} ---\\n\\n" + file.read()
        except Exception:
            pass
            
        if len(haystack) > num_chars:
            break
            
    # Cut to roughly num_chars
    haystack = haystack[:num_chars]
    
    # Inject needle
    inject_pos = int(len(haystack) * needle_pos_ratio)
    # Find nearest newline
    newline_pos = haystack.find('\\n', inject_pos)
    if newline_pos == -1:
        newline_pos = inject_pos
        
    final_text = haystack[:newline_pos] + f"\\n\\nUSER_SECRET_CODE_IS: {needle}\\n\\n" + haystack[newline_pos:]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as out:
        out.write(final_text)

print("Generating 20k char haystack (~5k tokens)...")
generate_realistic_haystack(".", "Tools/agent-eval/experiments/inputs/haystack_1k.txt", 20000, "ALPHA_PROTOCOL_ACTIVE", 0.75)

print("Generating 100k char haystack (~25k tokens)...")
generate_realistic_haystack(".", "Tools/agent-eval/experiments/inputs/haystack_10k.txt", 100000, "OMEGA_PROTOCOL_ACTIVE", 0.75)

print("Done!")
