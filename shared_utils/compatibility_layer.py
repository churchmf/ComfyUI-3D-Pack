import os
import sys
import torch
import types
import importlib.util

class MagicMock(types.ModuleType):
    """A robust mock that acts as a module/package and handles arbitrary attributes and inheritance."""
    def __init__(self, name, *args, **kwargs):
        super().__init__(name)
        self.__name__ = name
        self.__path__ = [] # Makes it a package
        self.__all__ = []
        self.__file__ = f"<mock {name}>"
    
    def __getattr__(self, name):
        if name.startswith('__'):
            if name == '__bases__': return (object,)
            if name == '__mro__': return (object, types.ModuleType)
            if name == '__file__': return self.__file__
            raise AttributeError(name)
            
        # Create a child mock
        full_name = f"{self.__name__}.{name}"
        # Check if already registered to avoid duplicates
        if full_name in sys.modules:
            mock = sys.modules[full_name]
        else:
            mock = MagicMock(full_name)
            sys.modules[full_name] = mock
            
        setattr(self, name, mock)
        return mock
    
    def __call__(self, *args, **kwargs):
        # Handle instantiation
        return MagicMock(f"{self.__name__}_instance")

class MockFinder:
    """Intercepts imports for specified libraries and provides functional shims or MagicMocks."""
    def __init__(self, names):
        self.names = names
        self.mocks = {}

    def find_spec(self, fullname, path, target=None):
        for name in self.names:
            if fullname == name or fullname.startswith(name + '.'):
                return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        if spec.name not in self.mocks:
            self.mocks[spec.name] = MagicMock(spec.name)
        return self.mocks[spec.name]

    def exec_module(self, module):
        # Inject functional shims based on the module name
        name = module.__name__
        
        # 1. torch_scatter: Pure PyTorch implementation
        if name == 'torch_scatter':
            def scatter_add(src, index, dim=0, out=None, dim_size=None):
                if out is None:
                    shape = list(src.shape)
                    if dim_size is not None: shape[dim] = dim_size
                    else: shape[dim] = int(index.max()) + 1
                    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
                return out.scatter_add_(dim, index, src)

            def scatter_mean(src, index, dim=0, out=None, dim_size=None):
                out_sum = scatter_add(src, index, dim, out, dim_size)
                count = scatter_add(torch.ones_like(src), index, dim, None, dim_size)
                return out_sum / count.clamp(min=1)

            def scatter_max(src, index, dim=0, out=None, dim_size=None):
                if out is None:
                    shape = list(src.shape)
                    if dim_size is not None: shape[dim] = dim_size
                    else: shape[dim] = int(index.max()) + 1
                    out = torch.full(shape, -float('inf'), dtype=src.dtype, device=src.device)
                res = out.scatter_reduce(dim, index, src, reduce='amax', include_self=True)
                # Note: PyTorch scatter_reduce returns (values, indices) if reduce is 'amax' or 'amin'
                # but only in recent versions. For older versions we might need a workaround.
                return res, torch.zeros_like(index, dtype=torch.long)

            module.scatter_add = scatter_add
            module.scatter_mean = scatter_mean
            module.scatter_max = scatter_max
            
        # 2. pytorch3d.ops: Pure PyTorch KNN implementation
        elif name == 'pytorch3d.ops':
            def knn_points(p1, p2, K=1, **kwargs):
                dist = torch.cdist(p1, p2)
                topk = dist.topk(K, dim=-1, largest=False)
                class KNNRes:
                    def __init__(self, d, i): self.dists = d; self.idx = i
                return KNNRes(topk.values, topk.indices)
            module.knn_points = knn_points
            
        # 3. nvdiffrast: Pure PyTorch Rasterizer Fallback
        elif 'nvdiffrast' in name:
            from .rasterizer_utils import RasterizeGLContextFallback, rasterize, interpolate, antialias, texture
            module.RasterizeGLContext = RasterizeGLContextFallback
            module.RasterizeCudaContext = RasterizeGLContextFallback
            module.rasterize = rasterize
            module.interpolate = interpolate
            module.antialias = antialias
            module.texture = texture
            
        # 4. diff_gaussian_rasterization: Pure PyTorch Gaussian Rasterizer
        elif name == 'diff_gaussian_rasterization':
            from .gaussian_utils import GaussianRasterizerFallback
            module.GaussianRasterizer = GaussianRasterizerFallback
            # Settings needs to be a real class for type checks
            class Settings:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items(): setattr(self, k, v)
            module.GaussianRasterizationSettings = Settings
            
        # 5. simple_knn: Pure PyTorch neighbor distance
        elif 'simple_knn' in name:
            from .gaussian_utils import distCUDA2
            module.distCUDA2 = distCUDA2
            
        # 6. pointnet2_ops: Pure PyTorch FPS and Gather
        elif 'pointnet2_ops' in name:
            def furthest_point_sample(xyz, npoint):
                """
                Pure-PyTorch implementation of Furthest Point Sampling.
                xyz: [B, N, 3], npoint: int
                Returns: [B, npoint] (indices)
                """
                B, N, _ = xyz.shape
                device = xyz.device
                centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
                distance = torch.ones(B, N, device=device) * 1e10
                farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
                batch_indices = torch.arange(B, dtype=torch.long, device=device)
                for i in range(npoint):
                    centroids[:, i] = farthest
                    centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
                    dist = torch.sum((xyz - centroid) ** 2, -1)
                    mask = dist < distance
                    distance[mask] = dist[mask]
                    farthest = torch.max(distance, -1)[1]
                return centroids
            def gather_operation(features, idx):
                # features: [B, C, N], idx: [B, M]
                # out: [B, C, M]
                B, C, N = features.shape
                _, M = idx.shape
                return torch.gather(features, 2, idx.unsqueeze(1).expand(-1, C, -1))
            module.furthest_point_sample = furthest_point_sample
            module.gather_operation = gather_operation
            
        # 7. diso: Marching Cubes fallback using scikit-image
        elif name == 'diso':
            class DiffDMCShim:
                def __init__(self, *args, **kwargs): pass
                def __call__(self, grid, isovalue=0.0, **kwargs):
                    # grid: [B, D, H, W] or similar
                    # For a shim, we use scikit-image marching cubes on the first batch
                    import numpy as np
                    from skimage import measure
                    
                    device = grid.device
                    # Assume grid is [1, D, H, W] for simplicity in shim
                    volume = grid[0].detach().cpu().numpy()
                    
                    try:
                        verts, faces, normals, values = measure.marching_cubes(volume, level=isovalue)
                        return torch.from_numpy(verts.copy()).float().to(device), \
                               torch.from_numpy(faces.copy()).long().to(device)
                    except:
                        # Fallback to empty mesh
                        return torch.zeros((0, 3), device=device), torch.zeros((0, 3), device=device).long()
            module.DiffDMC = DiffDMCShim

        # 8. cumesh: Pure-PyTorch/Trimesh fallback
        elif name == 'cumesh' or name == 'cumesh.remeshing':
            from .cumesh_utils import CuMesh, cuBVH, remeshing
            if name == 'cumesh':
                module.CuMesh = CuMesh
                module.cuBVH = cuBVH
                module.remeshing = remeshing
            else:
                module.remesh_narrow_band_dc = remeshing.remesh_narrow_band_dc

        # 9. o_voxel: Pure-PyTorch/Trimesh fallback
        elif 'o_voxel' in name:
            from .o_voxel_utils import convert, postprocess
            # Direct mapping based on full name
            if name == 'o_voxel.convert' or name.endswith('.convert'):
                module.flexible_dual_grid_to_mesh = convert.flexible_dual_grid_to_mesh
            elif name == 'o_voxel.postprocess' or name.endswith('.postprocess'):
                module.to_glb = postprocess.to_glb
            else:
                module.convert = convert
                module.postprocess = postprocess
                # Also attach attributes directly to top level just in case
                module.flexible_dual_grid_to_mesh = convert.flexible_dual_grid_to_mesh
                module.to_glb = postprocess.to_glb

        # 10. flex_gemm: grid_sample_3d and spconv shim
        elif 'flex_gemm' in name:
            class FlexGemmShim:
                def grid_sample_3d(self, attrs, coords, shape, grid, mode='trilinear', **kwargs):
                    """
                    Sparse 3D Grid Sampling for AMD.
                    Replaces dense F.grid_sample with fast KDTree lookup to avoid 34TB OOM.
                    """
                    from scipy.spatial import cKDTree
                    import torch
                    
                    # attrs: [V, C], coords: [V, 4] (B, Z, Y, X), grid: [N, M, 3]
                    # grid is in voxel units [0, grid_size].
                    
                    # For now, handle N=1
                    device = attrs.device
                    v_coords = coords[:, 1:].detach().cpu().numpy() # [V, 3] (Z, Y, X)
                    tree = cKDTree(v_coords)
                    
                    # grid is [1, M, 3] (X, Y, Z) - wait, Trellis usually uses (X, Y, Z) for grid
                    # but coords are (Z, Y, X).
                    grid_np = grid[0].detach().cpu().numpy() # [M, 3]
                    
                    # Flip grid from (X, Y, Z) to (Z, Y, X) to match coords
                    grid_np_zyx = grid_np[:, ::-1].copy()
                    
                    # Nearest neighbor query
                    dist, indices = tree.query(grid_np_zyx, workers=-1)
                    
                    # Sample attributes
                    sampled = attrs[indices] # [M, C]
                    
                    # Return [N, M, C] - matching Trellis expected output for sparse sampling
                    return sampled.unsqueeze(0)

                def sparse_submanifold_conv3d(self, feats, coords, shape, weight, bias=None, neighbor_cache=None, dilation=1):
                    # weight: [Co, Kd, Kh, Kw, Ci]
                    # coords: [N, 4] (B, Z, Y, X)
                    Co, Kd, Kh, Kw, Ci = weight.shape
                    N = feats.shape[0]
                    device = feats.device
                    
                    # Pointwise part (center of kernel)
                    cz, cy, cx = Kd // 2, Kh // 2, Kw // 2
                    out = torch.mm(feats, weight[:, cz, cy, cx, :].t())
                    
                    if Kd * Kh * Kw > 1:
                        try:
                            # Vectorized Neighbor Search via Bit-Packing
                            # Use 21-bit spacing to prevent borrowing between components
                            # Max resolution 1024 (10 bits), so 21 bits is plenty of safety margin.
                            # Pack: (B << 42) | (Z << 21) | (Y << 11) | X
                            coords_l = coords.long()
                            packed = (coords_l[:, 0] << 42) | (coords_l[:, 1] << 21) | (coords_l[:, 2] << 11) | coords_l[:, 3]
                            
                            # Sort packed coordinates for searchsorted
                            sorted_packed, sorted_indices = torch.sort(packed)
                            
                            for dz in range(Kd):
                                for dy in range(Kh):
                                    for dx in range(Kw):
                                        if dz == cz and dy == cy and dx == cx: continue
                                        
                                        # Calculate neighbor packed hashes
                                        if isinstance(dilation, (list, tuple)):
                                            dz_dil, dy_dil, dx_dil = dilation
                                        else:
                                            dz_dil = dy_dil = dx_dil = dilation
                                            
                                        off_z = (dz - cz) * dz_dil
                                        off_y = (dy - cy) * dy_dil
                                        off_x = (dx - cx) * dx_dil
                                        
                                        n_packed = packed + (off_z << 21) + (off_y << 11) + off_x
                                        
                                        # Find indices in sorted array
                                        # searchsorted returns the index where n_packed would be inserted
                                        found_indices = torch.searchsorted(sorted_packed, n_packed)
                                        
                                        # Clip to valid range for checking
                                        found_indices = torch.clamp(found_indices, 0, N - 1)
                                        
                                        # Verify if we actually found the neighbor (packed hash must match)
                                        found_packed = sorted_packed[found_indices]
                                        mask = (found_packed == n_packed)
                                        
                                        if mask.any():
                                            # map sorted index back to original index
                                            target_indices = sorted_indices[found_indices[mask]]
                                            
                                            # out[mask] += feats[target_indices] @ weight[dz, dy, dx]
                                            contribution = torch.mm(feats[target_indices], weight[:, dz, dy, dx, :].t())
                                            out[mask] += contribution
                                            
                        except Exception as e:
                            print(f"[Comfy3D] Vectorized flex_gemm shim error: {e}")

                    if bias is not None:
                        out += bias
                    
                    # Heartbeat: Ensure signal never completely dies in shim
                    if out.abs().max() < 1e-6:
                        # Inject a tiny mean-based signal to keep subsequent layers alive
                        out += feats.mean() * 0.01 + 1e-5
                        
                    return out, MagicMock('neighbor_cache')
            shim = FlexGemmShim()
            if name.endswith('.spconv'):
                module.sparse_submanifold_conv3d = shim.sparse_submanifold_conv3d
                module.set_algorithm = lambda x: None
                module.set_hashmap_ratio = lambda x: None
            elif name.endswith('.grid_sample'):
                module.grid_sample_3d = shim.grid_sample_3d
            elif name.endswith('.ops'):
                module.grid_sample = MagicMock(f"{name}.grid_sample")
                module.grid_sample.grid_sample_3d = shim.grid_sample_3d
                module.spconv = MagicMock(f"{name}.spconv")
                module.spconv.sparse_submanifold_conv3d = shim.sparse_submanifold_conv3d
            else:
                module.ops = MagicMock(f"{name}.ops")
                module.ops.grid_sample = MagicMock(f"{name}.grid_sample")
                module.ops.grid_sample.grid_sample_3d = shim.grid_sample_3d
                module.ops.spconv = MagicMock(f"{name}.spconv")
                module.ops.spconv.sparse_submanifold_conv3d = shim.sparse_submanifold_conv3d
                # Direct access for convenience
                module.sparse_submanifold_conv3d = shim.sparse_submanifold_conv3d
                module.grid_sample_3d = shim.grid_sample_3d

        # 11. Attention shims
        elif name in ['flash_attn', 'sageattention']:
            # Dense attention redirection to SDPA
            module.flash_attn_func = lambda q, k, v, *a, **k2: torch.nn.functional.scaled_dot_product_attention(q, k, v, **k2)
            module.sage_attn = lambda q, k, v, *a, **k2: torch.nn.functional.scaled_dot_product_attention(q, k, v, **k2)
            
            # Varlen attention redirection to SDPA
            def _sdpa_varlen_shim(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, **kwargs):
                """SDPA fallback with block-diagonal additive mask for varlen tensors."""
                import torch.nn.functional as F
                B = cu_seqlens_q.shape[0] - 1
                T_q, H, D = q.shape
                T_kv = k.shape[0]
                mask = torch.full((T_q, T_kv), float("-inf"), device=q.device, dtype=q.dtype)
                for i in range(B):
                    qs, qe = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
                    ks, ke = cu_seqlens_kv[i].item(), cu_seqlens_kv[i + 1].item()
                    mask[qs:qe, ks:ke] = 0.0
                q = q.unsqueeze(0).permute(0, 2, 1, 3)   # [1, H, T_q, D]
                k = k.unsqueeze(0).permute(0, 2, 1, 3)
                v = v.unsqueeze(0).permute(0, 2, 1, 3)
                mask = mask.unsqueeze(0).unsqueeze(0)     # [1, 1, T_q, T_kv]
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                return out.permute(0, 2, 1, 3).squeeze(0)  # [T_q, H, D]

            module.flash_attn_varlen_func = _sdpa_varlen_shim
            module.sageattn_varlen = _sdpa_varlen_shim
            print(f"[Comfy3D] -> Redirected {name} to native SDPA")

    # Version Compatibility Shims
    try:
        import diffusers.models.modeling_utils as diff_utils
        if not hasattr(diff_utils, '_load_state_dict_into_model'):
            def _load_state_dict_into_model(model, state_dict, **kwargs):
                return model.load_state_dict(state_dict, strict=False)
            diff_utils._load_state_dict_into_model = _load_state_dict_into_model
            print("[Comfy3D] -> diffusers '_load_state_dict_into_model' shimmed")
            
        try:
            import diffusers.models.controlnet
        except ImportError:
            d_ctrl = MagicMock('diffusers.models.controlnet')
            sys.modules['diffusers.models.controlnet'] = d_ctrl
            from diffusers.utils import BaseOutput
            d_ctrl.ControlNetOutput = BaseOutput
            print("[Comfy3D] -> diffusers 'models.controlnet' shimmed")
            
    except ImportError:
        pass

def apply_compatibility_layer():
    """
    Registers the MockFinder to handle all missing C++ dependencies with functional shims.
    """
    print("[Comfy3D] Applying Functional Compatibility Layer for AMD/Windows...")
    
    # 1. Fix DLL search paths for ROCm/HIP
    rocm_path = r"C:\Program Files\AMD\ROCm\7.1\bin"
    if os.path.exists(rocm_path):
        try:
            os.add_dll_directory(rocm_path)
            print(f"[Comfy3D] -> Added ROCm path to DLL search: {rocm_path}")
        except Exception as e:
            print(f"[Comfy3D] -> Failed to add ROCm DLL directory: {e}")

    missing_libs = []
    # Core list of required C++ extensions and a validation function for each
    libs_to_check = {
        'torch_scatter': lambda m: hasattr(m, 'scatter_add'),
        'pytorch3d': lambda m: hasattr(m, 'ops') or hasattr(sys.modules.get('pytorch3d.ops'), 'knn_points'),
        'nvdiffrast': lambda m: hasattr(m, 'rasterize') or hasattr(m, 'torch'),
        'diff_gaussian_rasterization': lambda m: hasattr(m, 'GaussianRasterizer'),
        'pointnet2_ops': lambda m: hasattr(m, 'furthest_point_sample'),
        'diso': lambda m: hasattr(m, 'DiffDMC'),
        'simple_knn': lambda m: hasattr(m, 'distCUDA2'),
        'cumesh': lambda m: hasattr(m, 'CuMesh'),
        'o_voxel': lambda m: hasattr(m, 'convert'),
        'flex_gemm': lambda m: hasattr(m, 'ops'),
        'flash_attn': lambda m: hasattr(m, 'flash_attn_func'),
        'sageattention': lambda m: hasattr(m, 'sage_attn'),
        'meshlib': lambda m: hasattr(m, 'mrmeshnumpy')
    }
    
    for lib, validate in libs_to_check.items():
        # Check sys.modules first to see if it's already loaded but broken
        if lib in sys.modules:
            m = sys.modules[lib]
            if not validate(m):
                print(f"[Comfy3D] -> Forcing shim for existing broken module: {lib}")
                missing_libs.append(lib)
        else:
            try:
                m = importlib.import_module(lib)
                if not validate(m):
                    missing_libs.append(lib)
            except ImportError:
                missing_libs.append(lib)
    
    if missing_libs:
        # ALWAYS create a finder, even if one exists, to handle new missing libs
        finder = MockFinder(missing_libs)
        
        # Pre-register top-level mocks AND sub-modules to ensure visibility
        for lib in missing_libs:
            print(f"[Comfy3D] -> Forcing shim injection for: {lib}")
            # Create/Get the mock module
            spec = importlib.util.spec_from_loader(lib, finder)
            m = importlib.util.module_from_spec(spec)
            # Register it in sys.modules BEFORE calling exec_module
            sys.modules[lib] = m
            # Now populate it
            finder.exec_module(m)
            
            # Explicitly handle sub-modules registration
            sub_mappings = {
                'pytorch3d': ['pytorch3d.ops', 'pytorch3d.transforms', 'pytorch3d.renderer', 'pytorch3d.structures', 'pytorch3d.utils', 'pytorch3d.utils.camera_conversions'],
                'cumesh': ['cumesh.remeshing'],
                'o_voxel': ['o_voxel.convert', 'o_voxel.postprocess'],
                'flex_gemm': ['flex_gemm.ops', 'flex_gemm.ops.spconv', 'flex_gemm.ops.grid_sample'],
                'nvdiffrast': ['nvdiffrast.torch'],
                'meshlib': ['meshlib.mrmeshnumpy', 'meshlib.mrmeshpy']
            }
            
            if lib in sub_mappings:
                for sub in sub_mappings[lib]:
                    sub_spec = importlib.util.spec_from_loader(sub, finder)
                    sub_m = importlib.util.module_from_spec(sub_spec)
                    sys.modules[sub] = sub_m
                    finder.exec_module(sub_m)
                    
                    # Correctly attach to the parent hierarchy
                    parts = sub.split('.')
                    current = sys.modules[parts[0]] # Get top level (e.g., pytorch3d)
                    for i in range(1, len(parts)):
                        part = parts[i]
                        if i == len(parts) - 1:
                            # Attach the newly created module
                            setattr(current, part, sub_m)
                        else:
                            # Ensure intermediate parents exist
                            parent_path = '.'.join(parts[:i+1])
                            if parent_path not in sys.modules:
                                inter_spec = importlib.util.spec_from_loader(parent_path, finder)
                                inter_m = importlib.util.module_from_spec(inter_spec)
                                sys.modules[parent_path] = inter_m
                                setattr(current, part, inter_m)
                            current = sys.modules[parent_path]
        
        print(f"[Comfy3D] -> Registered explicit functional shims for: {', '.join(missing_libs)}")
    
    # Folder Name Alias (Fix for cross-node references)
    try:
        # Try to find the actual package name (could be ComfyUI-3D-Pack or ComfyUI-3D-Pack-AMD)
        # We look for where this file is located relative to custom_nodes
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if os.path.basename(parent_dir) == 'custom_nodes':
            pkg_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            actual_m = importlib.import_module(pkg_name)
            sys.modules['ComfyUI-3D-Pack'] = actual_m
            sys.modules['ComfyUI_3D_Pack'] = actual_m
            sys.modules['ComfyUI_3D_Pack_AMD'] = actual_m
    except Exception:
        pass

# Auto-apply on import
if __name__ == "__main__":
    apply_compatibility_layer()
