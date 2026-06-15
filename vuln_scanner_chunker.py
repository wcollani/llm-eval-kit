#!/usr/bin/env python3
import os
import glob
import json

def get_largest_files(repo_path, num_files=20):
    files = []
    for root, _, filenames in os.walk(repo_path):
        if '.git' in root or '.venv' in root:
            continue
        for filename in filenames:
            if not filename.endswith(('.py', '.go', '.md', '.json', '.yaml', '.sh')):
                continue
            filepath = os.path.join(root, filename)
            try:
                files.append((filepath, os.path.getsize(filepath)))
            except OSError:
                pass
    files.sort(key=lambda x: x[1], reverse=True)
    return files[:num_files]

def chunk_file(filepath, chunk_size=4000):
    chunks = []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            for i in range(0, len(content), chunk_size):
                chunks.append(content[i:i+chunk_size])
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return chunks

def main():
    repo_path = os.path.abspath("../../") # Assuming running from Tools/agent-eval
    largest_files = get_largest_files(repo_path)
    
    os.makedirs("inputs/vuln_chunks", exist_ok=True)
    
    chunk_index = 0
    manifest = []
    for filepath, size in largest_files:
        chunks = chunk_file(filepath)
        for chunk in chunks:
            out_file = f"inputs/vuln_chunks/chunk_{chunk_index}.txt"
            with open(out_file, "w") as f:
                f.write(chunk)
            manifest.append({"file": filepath, "chunk_file": out_file})
            chunk_index += 1
            
    with open("inputs/vuln_chunks/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Generated {chunk_index} chunks from the top {len(largest_files)} largest files.")

if __name__ == "__main__":
    main()
