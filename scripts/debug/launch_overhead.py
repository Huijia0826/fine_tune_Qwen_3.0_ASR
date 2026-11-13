"""E7 profiling debug (2026-11-02): is the HF decoder step CPU/launch bound on this node? Node load, kernel-launch
cost, and one decoder step's CPU dispatch time vs wall time (if equal, the GPU waits for the CPU)."""
import os
import time

import torch

print("cpus allowed", len(os.sched_getaffinity(0)), "loadavg", os.getloadavg(), "torch threads", torch.get_num_threads())
x = torch.zeros(16, device="cuda")
for _ in range(100):
    x.add_(1)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10000):
    x.add_(1)
t_cpu = time.perf_counter() - t0
torch.cuda.synchronize()
t_all = time.perf_counter() - t0
print(f"tiny kernel launch: {1e6 * t_cpu / 10000:.1f} us/launch (CPU side), {1e6 * t_all / 10000:.1f} us incl. GPU")

from whisperflow.data.common import load_yaml  # noqa: E402
from whisperflow.eval.transcribe import HFWhisperTranscriber  # noqa: E402

cfg = load_yaml("configs/experiments/latency/hf_bf16_beam5.yaml")
tr = HFWhisperTranscriber.from_config(cfg["model"], cfg["decoding"], cfg["runtime"])
m, dec = tr.model, tr.model.model.decoder
enc = torch.randn(1, 1500, m.config.d_model, device="cuda", dtype=torch.bfloat16)
with torch.inference_mode():
    cache = dec(input_ids=torch.randint(0, 50000, (1, 40), device="cuda"), encoder_hidden_states=enc,
                use_cache=True).past_key_values
    nxt = torch.randint(0, 50000, (1, 1), device="cuda")
    for i in range(8):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m.proj_out(dec(input_ids=nxt, encoder_hidden_states=enc, past_key_values=cache, use_cache=True).last_hidden_state)
        t_cpu = time.perf_counter() - t0
        torch.cuda.synchronize()
        t_all = time.perf_counter() - t0
        print(f"decoder step {i}: CPU dispatch {1e3 * t_cpu:.1f} ms, wall {1e3 * t_all:.1f} ms")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        m.proj_out(dec(input_ids=nxt, encoder_hidden_states=enc, past_key_values=cache, use_cache=True).last_hidden_state)
        torch.cuda.synchronize()
ev = prof.key_averages()
gpu_us = sum(e.self_device_time_total for e in ev)
n_kernels = sum(e.count for e in ev if e.self_device_time_total > 0)
print(f"profiled step: GPU busy {gpu_us / 1e3:.2f} ms over ~{n_kernels} GPU ops")
print(ev.table(sort_by="self_cpu_time_total", row_limit=8))
