"""Verify CUDA 4-bit forward, backward and optimizer execution without weights."""

import json

import bitsandbytes as bnb
import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback.")
    torch.cuda.reset_peak_memory_stats()
    layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.float16,
                              compress_statistics=True, quant_type="nf4").to("cuda")
    adapter = torch.nn.Parameter(torch.randn(64, 4, device="cuda", dtype=torch.float32))
    x = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    optimizer = torch.optim.AdamW([adapter], lr=0.001)
    before = adapter.detach().clone()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = (layer(x).float() + (x.float() @ adapter).mean()).square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or torch.equal(before, adapter):
        raise RuntimeError("4-bit GPU optimizer probe failed.")
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                      "bitsandbytes": bnb.__version__, "gpu": torch.cuda.get_device_name(),
                      "loss_finite": True, "adapter_changed": True,
                      "free_bytes": free, "total_bytes": total,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated()}, indent=2))


if __name__ == "__main__":
    main()
