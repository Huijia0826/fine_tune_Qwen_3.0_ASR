"""E7 profiling debug (2026-11-02): decoder-step time per SDPA backend (cuDNN showed ~1.6 ms CPU per attention call)."""
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from whisperflow.data.common import load_yaml
from whisperflow.eval.transcribe import HFWhisperTranscriber

cfg = load_yaml("configs/experiments/latency/hf_bf16_beam5.yaml")
tr = HFWhisperTranscriber.from_config(cfg["model"], cfg["decoding"], cfg["runtime"])
m, dec = tr.model, tr.model.model.decoder
print("attn_implementation", m.config._attn_implementation)
backends = {"cudnn": SDPBackend.CUDNN_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}
for name, be in backends.items():
    for b in (1, 5):
        enc = torch.randn(b, 1500, m.config.d_model, device="cuda", dtype=torch.bfloat16)
        try:
            with torch.inference_mode(), sdpa_kernel([be]):
                cache = dec(input_ids=torch.randint(0, 50000, (b, 40), device="cuda"), encoder_hidden_states=enc,
                            use_cache=True).past_key_values
                nxt = torch.randint(0, 50000, (b, 1), device="cuda")
                ts = []
                for i in range(23):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    m.proj_out(dec(input_ids=nxt, encoder_hidden_states=enc, past_key_values=cache,
                                   use_cache=True).last_hidden_state)
                    torch.cuda.synchronize()
                    if i >= 3:
                        ts.append(time.perf_counter() - t0)
            ts.sort()
            print(f"{name:9s} batch {b}: decoder step median {1e3 * ts[len(ts) // 2]:.2f} ms")
        except Exception as e:
            print(f"{name:9s} batch {b}: {type(e).__name__} {str(e)[:100]}")
