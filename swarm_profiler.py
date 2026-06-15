#!/usr/import/env python3
import asyncio
import time
import os
import json
import argparse
import litellm
from datetime import datetime
import numpy as np

# Suppress debug logs for cleaner output
litellm.suppress_debug_info = True

async def single_request(model: str, prompt: str, timeout: int = 600):
    start_time = time.time()
    try:
        api_base = os.getenv("LITELLM_API_BASE", "http://localhost:4000/v1")
        if model.startswith("ws/"):
            api_base = os.getenv("OLLAMA_WS_URL", "http://localhost:11434/v1")
            model = model[3:]
        elif model.startswith("sre/"):
            api_base = os.getenv("OLLAMA_SRE_URL", "http://localhost:11434/v1")
            model = model[4:]
            
        if model.startswith("ollama/"):
            model = model[7:]
            
        proxy_model = f"openai/{model}" if not model.startswith("openai/") else model
        
        res = await asyncio.wait_for(
            litellm.acompletion(
                model=proxy_model,
                messages=[{"role": "user", "content": prompt}],
                api_base=api_base,
                api_key="sk-dummy",
                num_ctx=8192,
                timeout=timeout
            ),
            timeout=timeout
        )
        latency = time.time() - start_time
        return {"success": True, "latency": latency, "error": None}
    except Exception as e:
        latency = time.time() - start_time
        return {"success": False, "latency": latency, "error": repr(e)}

async def run_batch(model: str, concurrency: int, prompt: str):
    print(f"\n[+] Blasting {concurrency} parallel requests to {model}...")
    start_time = time.time()
    
    tasks = [single_request(model, prompt) for _ in range(concurrency)]
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    
    latencies = [r["latency"] for r in successes]
    avg_latency = np.mean(latencies) if latencies else 0
    p95_latency = np.percentile(latencies, 95) if latencies else 0
    
    rps = len(successes) / total_time if total_time > 0 else 0
    
    print(f"    - Completed in {total_time:.2f}s")
    print(f"    - Success: {len(successes)} | Failures: {len(failures)}")
    print(f"    - RPS: {rps:.2f} req/s")
    if latencies:
        print(f"    - Avg Latency: {avg_latency:.2f}s | p95 Latency: {p95_latency:.2f}s")
    if failures:
        # Just grab the first failure reason for logging
        print(f"    - Example Error: {failures[0]['error']}")
        
    return {
        "concurrency": concurrency,
        "total_time": round(total_time, 2),
        "success_count": len(successes),
        "failure_count": len(failures),
        "rps": round(rps, 2),
        "avg_latency": round(avg_latency, 2),
        "p95_latency": round(p95_latency, 2)
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="ollama/qwen2.5-coder:7b,ollama/qwen2.5-coder:14b", help="Comma separated models")
    parser.add_argument("--levels", type=str, default="5,10,25,50", help="Comma separated concurrency levels")
    parser.add_argument("--payload-size", type=int, default=0, help="Number of dummy words to append to the prompt to test context OOM")
    args = parser.parse_args()
    
    models = [m.strip() for m in args.models.split(",")]
    levels = [int(l.strip()) for l in args.levels.split(",")]
    
    prompt = "You are a monitoring subagent. Return exactly this PromQL query and nothing else: `container_cpu_usage_seconds_total{container=\"api-server\"}`"
    if args.payload_size > 0:
        prompt += "\nHere is additional context:\n" + ("dummy_token " * args.payload_size)    
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "runs": {}
    }
    
    for model in models:
        final_results["runs"][model] = []
        for level in levels:
            res = await run_batch(model, level, prompt)
            final_results["runs"][model].append(res)
            # Cool down to let GPU flush queue
            if level != levels[-1]:
                print("    [Cooling down for 5 seconds...]")
                await asyncio.sleep(5)
                
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    out_file = f"results/Swarm_Concurrency_Profile_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\n[*] Profiling complete! Results saved to {out_file}")

if __name__ == "__main__":
    asyncio.run(main())
