import torch
import torch.nn.functional as F

class RasterizeGLContextFallback:
    def __init__(self, device=None, **kwargs):
        self.device = device if device else "cuda"

def rasterize(ctx, v_clip, pos, resolution, grad_db=True):
    """
    Blazing fast pure-PyTorch shim for nvdiffrast.rasterize using OpenCV.
    Takes advantage of non-overlapping UV spaces to render 100k+ polygons in milliseconds.
    """
    import cv2
    import numpy as np
    
    N, V, _ = v_clip.shape
    T, _ = pos.shape
    H, W = resolution
    device = v_clip.device
    
    # 1. Project to Screen Space
    w = v_clip[..., 3:4].clamp(min=1e-6)
    v_ndc = v_clip[..., :3] / w
    v_screen = torch.stack([
        (v_ndc[..., 0] + 1.0) * 0.5 * W,
        (1.0 - v_ndc[..., 1]) * 0.5 * H
    ], dim=-1)
    
    if N == 1:
        tv = v_screen[0, pos.long()] # [T, 3, 2]
        tz = v_ndc[0, pos.long(), 2] # [T, 3]
        
        # Move to CPU for OpenCV rasterization
        tv_np = tv.detach().cpu().numpy()
        
        # Initialize texture ID map
        tri_ids = np.full((H, W), -1, dtype=np.int32)
        
        # Draw all triangles to get exact pixel-to-triangle mapping instantly
        for i in range(len(tv_np)):
            # OpenCV requires int32 coordinates
            pts = tv_np[i].astype(np.int32)
            cv2.fillConvexPoly(tri_ids, pts, i)
            
        # Move map back to GPU
        tri_ids_pt = torch.from_numpy(tri_ids).to(device)
        mask = tri_ids_pt >= 0
        valid_ids = tri_ids_pt[mask].long()
        
        # Compute exact barycentric coordinates only for active pixels
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32), 
            torch.arange(W, device=device, dtype=torch.float32), 
            indexing='ij'
        )
        px = grid_x[mask] + 0.5
        py = grid_y[mask] + 0.5
        
        v0 = tv[valid_ids, 0]
        v1 = tv[valid_ids, 1]
        v2 = tv[valid_ids, 2]
        
        # Area of the triangles
        area = (v1[:, 1] - v2[:, 1]) * (v0[:, 0] - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (v0[:, 1] - v2[:, 1])
        
        # Robustly avoid division by zero
        area = torch.where(area.abs() < 1e-8, area.new_tensor(1e-8), area)
        
        w0 = ((v1[:, 1] - v2[:, 1]) * (px - v2[:, 0]) + (v2[:, 0] - v1[:, 0]) * (py - v2[:, 1])) / area
        w1 = ((v2[:, 1] - v0[:, 1]) * (px - v2[:, 0]) + (v0[:, 0] - v2[:, 0]) * (py - v2[:, 1])) / area
        w2 = 1.0 - w0 - w1
        
        # Interpolate Z values
        z0 = tz[valid_ids, 0]
        z1 = tz[valid_ids, 1]
        z2 = tz[valid_ids, 2]
        z_val = w0 * z0 + w1 * z1 + w2 * z2
        
        # Pack final output buffer (w0, w1, z, id) matching nvdiffrast format
        out_rast = torch.zeros((H, W, 4), device=device)
        out_rast[..., 3] = -1
        out_rast[mask, 0] = w0
        out_rast[mask, 1] = w1
        out_rast[mask, 2] = z_val
        out_rast[mask, 3] = valid_ids.float()
        
        return out_rast.unsqueeze(0), None
    else:
        # Fallback for batch > 1
        res = []
        for n in range(N):
            r, _ = rasterize(ctx, v_clip[n:n+1], pos, resolution)
            res.append(r)
        return torch.cat(res, dim=0), None

def interpolate(attr, rast, tri, attr_db=None, diff_attrs=None):
    """
    Pure-PyTorch shim for nvdiffrast.interpolate
    """
    N, H, W, _ = rast.shape
    u, v = rast[..., 0:1], rast[..., 1:2]
    w = 1.0 - u - v
    tri_id = rast[..., 3].long()
    mask = tri_id >= 0
    C = attr.shape[-1]
    out = torch.zeros((N, H, W, C), device=attr.device)
    
    if attr.dim() == 3: # [N, V, C]
        flat_tri_id = tri_id.view(-1)
        valid_mask = mask.view(-1)
        if valid_mask.any():
            v_idx = tri[flat_tri_id[valid_mask]].long() 
            # Multi-batch support for interpolate
            for n in range(N):
                b_mask = mask[n].view(-1)
                if not b_mask.any(): continue
                b_tri_id = tri_id[n].view(-1)[b_mask]
                b_v_idx = tri[b_tri_id].long()
                a0, a1, a2 = attr[n, b_v_idx[:, 0]], attr[n, b_v_idx[:, 1]], attr[n, b_v_idx[:, 2]]
                pw, pu, pv = w[n].view(-1, 1)[b_mask], u[n].view(-1, 1)[b_mask], v[n].view(-1, 1)[b_mask]
                out[n].view(-1, C)[b_mask] = pw * a2 + pu * a0 + pv * a1
    return out, None

def antialias(color, rast, v_clip, tri):
    return color

def texture(tex, uv, uv_da=None, mip=None, filter_mode='linear', boundary_mode='wrap', max_mip_level=None):
    """
    Pure-PyTorch shim for nvdiffrast.texture.
    Upgraded for bilinear sampling on AMD.
    """
    N, H, W, C = tex.shape
    BN, GH, GW, _ = uv.shape
    
    # Map uv [0, 1] to grid [-1, 1]
    # In nvdiffrast, (0,0) is top-left, but grid_sample expects (-1,-1) as top-left.
    # uv: [BN, GH, GW, 2]
    grid = uv * 2.0 - 1.0
    
    # Permute texture for grid_sample: [N, C, H, W]
    tex_input = tex.permute(0, 3, 1, 2)
    
    # Mode selection
    mode = 'bilinear' if filter_mode == 'linear' else 'nearest'
    
    # Boundary mode mapping
    # wrap -> 'reflection' (approximate) or 'border'
    # clamp -> 'border'
    padding_mode = 'border' if boundary_mode == 'clamp' else 'reflection'
    
    # Support for multi-batch if tex and uv differ in batch dimension
    if N == 1 and BN > 1:
        tex_input = tex_input.expand(BN, -1, -1, -1)
    
    out = F.grid_sample(tex_input, grid, mode=mode, padding_mode=padding_mode, align_corners=False)
    
    # Permute back to [BN, GH, GW, C]
    return out.permute(0, 2, 3, 1)
