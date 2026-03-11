import torch
import os

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # For DirectML if installed later
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")

def is_amd():
    if not torch.cuda.is_available():
        return True
    # ROCm identifies as CUDA but usually contains 'roc' or 'hip' in version
    if hasattr(torch, "version") and (hasattr(torch.version, "hip") or "roc" in torch.__version__):
        return True
    return False

_LOGGED_AMD = False

def get_rasterize_context(device=None):
    global _LOGGED_AMD
    import nvdiffrast.torch as dr
    
    # On AMD/ROCm, RasterizeCudaContext usually fails even if torch.cuda.is_available() is True
    # because it specifically looks for NVIDIA CUDA libraries.
    if is_amd():
        if not _LOGGED_AMD:
            print("[Comfy3D] AMD/ROCm detected, using RasterizeGLContext")
            _LOGGED_AMD = True
        return dr.RasterizeGLContext(device=device)
    
    try:
        return dr.RasterizeCudaContext(device=device)
    except Exception as e:
        print(f"[Comfy3D] RasterizeCudaContext failed, falling back to RasterizeGLContext: {e}")
        return dr.RasterizeGLContext(device=device)
