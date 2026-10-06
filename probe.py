"""Read-only environment and CUDA computation probe."""
import importlib
import json
import sys

report = {"python": sys.version, "executable": sys.executable}
for name in ["torch", "torchvision", "numpy", "scipy", "sklearn", "PIL", "matplotlib", "cv2", "onnxruntime", "timm", "requests"]:
    try:
        module = importlib.import_module(name)
        report[name] = getattr(module, "__version__", "installed")
    except Exception as exc:
        report[name] = str(exc)
try:
    import torch
    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_version"] = torch.version.cuda
    report["device"] = torch.cuda.get_device_name(0)
    x = torch.randn(512, 512, device="cuda", requires_grad=True)
    y = (x @ x).square().mean()
    y.backward()
    torch.cuda.synchronize()
    report["cuda_forward_backward"] = bool(torch.isfinite(x.grad).all().item())
except Exception as exc:
    report["cuda_error"] = repr(exc)
print(json.dumps(report, indent=2))
