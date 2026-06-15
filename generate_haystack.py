import os
import random

def generate_haystack(filename, num_lines, needle, needle_line):
    words = ["homelab", "docker", "kubernetes", "vllm", "ollama", "latency", "node", "cluster", "gpu", "vram", "metrics", "grafana", "unraid", "network", "firewall", "storage", "nvme"]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        for i in range(num_lines):
            if i == needle_line:
                f.write(f"USER_SECRET_CODE_IS: {needle}\n")
            else:
                line_words = [random.choice(words) for _ in range(20)]
                f.write(" ".join(line_words) + "\n")

generate_haystack("Tools/agent-eval/experiments/inputs/haystack_10k.txt", 10000, "OMEGA_PROTOCOL_ACTIVE", 7500)
generate_haystack("Tools/agent-eval/experiments/inputs/haystack_1k.txt", 1000, "ALPHA_PROTOCOL_ACTIVE", 750)
