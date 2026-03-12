import torch
import numpy as np
import trimesh
from skimage import measure
from scipy.ndimage import binary_dilation

def flexible_dual_grid_to_mesh(*args, **kwargs):
    """
    Robust solid mesh reconstruction using trimesh.voxel and scipy dilation.
    """
    if len(args) >= 3:
        coords = args[0]
        v_feats = args[1]
        i_feats = args[2]
    else:
        coords = kwargs.get('coords')
        v_feats = kwargs.get('dual_vertices', kwargs.get('v_feats'))
        i_feats = kwargs.get('intersected_flag', kwargs.get('i_feats'))
        
    if coords is None or i_feats is None:
        return torch.zeros((0, 3)), torch.zeros((0, 3)).long()
    
    device = coords.device
    
    try:
        # 1. Prepare Data
        c_raw = coords.detach().cpu().numpy().astype(np.float32)
        i_f = i_feats.detach().cpu().float().numpy()
        
        if c_raw.shape[1] == 4:
            c = c_raw[:, 1:] 
        else:
            c = c_raw
            
        c_max = c.max()
        if c_max <= 32: res = 32
        elif c_max <= 64: res = 64
        elif c_max <= 128: res = 128
        elif c_max <= 256: res = 256
        else: res = 512
        
        # 2. Extract Occupancy
        occ = i_f.max(axis=-1)
        mask = occ > 0.5
        if not np.any(mask):
            return torch.zeros((0, 3), device=device), torch.zeros((0, 3), device=device).long()
            
        # 3. Create Voxel Grid
        min_c = c.min(axis=0).astype(np.int64)
        max_c = c.max(axis=0).astype(np.int64)
        shape = max_c - min_c + 3 # Add margin
        shape = np.clip(shape, 1, 256)
        
        encoding = np.zeros(shape, dtype=bool)
        c_local = (c[mask] - min_c).astype(np.int64) + 1
        valid_local = np.all((c_local >= 0) & (c_local < shape), axis=1)
        encoding[c_local[valid_local, 0], c_local[valid_local, 1], c_local[valid_local, 2]] = True
        
        # 4. Dilate for connectivity
        encoding = binary_dilation(encoding, iterations=1)
        
        # 5. Marching Cubes via VoxelGrid
        vox = trimesh.voxel.VoxelGrid(encoding)
        mesh = vox.marching_cubes
        
        # 6. Transform back
        verts = (mesh.vertices - 1 + min_c) / res - 0.5
        # Flip ZYX to XYZ
        verts = verts[:, ::-1].copy()
        
        return torch.from_numpy(verts).float().to(device), \
               torch.from_numpy(mesh.faces).long().to(device)

    except Exception as e:
        print(f"[Comfy3D] FDG Trimesh Voxel Error: {e}")
        return torch.zeros((0, 3), device=device), torch.zeros((0, 3), device=device).long()

def tiled_flexible_dual_grid_to_mesh(*args, **kwargs):
    return flexible_dual_grid_to_mesh(*args, **kwargs)

def to_glb(vertices, faces, attr_volume, coords, attr_layout, **kwargs):
    import trimesh
    import torch
    import sys
    try:
        print("[Comfy3D] to_glb: Starting texture baking pipeline...")
        import nodes
        Trellis2Simplify = nodes.NODE_CLASS_MAPPINGS["Trellis2Simplify"]
        Trellis2UVUnwrap = nodes.NODE_CLASS_MAPPINGS["Trellis2UVUnwrap"]
        Trellis2RasterizePBR = nodes.NODE_CLASS_MAPPINGS["Trellis2RasterizePBR"]
        if torch.is_tensor(vertices): v = vertices.detach().cpu().numpy()
        else: v = vertices
        if torch.is_tensor(faces): f = faces.detach().cpu().numpy()
        else: f = faces
        if v.shape[0] == 0:
            return trimesh.Trimesh(vertices=np.zeros((0,3)), faces=np.zeros((0,3)))
        base_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        print(f"[Comfy3D] to_glb: Base mesh created ({len(v)} verts, {len(f)} faces)")
        decimation_target = kwargs.get('decimation_target', 50000)
        remesh = kwargs.get('remesh', True)
        simplifier = Trellis2Simplify()
        simplified_mesh = simplifier.simplify(base_mesh, target_face_count=decimation_target, remesh=remesh)[0]
        unwrapper = Trellis2UVUnwrap()
        unwrapped_mesh = unwrapper.unwrap(simplified_mesh)[0]
        rasterizer = Trellis2RasterizePBR()
        voxelgrid = {'coords': coords, 'attrs': attr_volume, 'layout': attr_layout, 'voxel_size': 1.0, 'original_vertices': v, 'original_faces': f}
        textured_mesh = rasterizer.rasterize(unwrapped_mesh, voxelgrid, texture_size=1024)[0]
        print("[Comfy3D] to_glb: Texture baking successful!")
        return textured_mesh
    except Exception as e:
        print(f"[Comfy3D] to_glb failure: {e}")
        if torch.is_tensor(vertices): v = vertices.detach().cpu().numpy()
        else: v = vertices
        if torch.is_tensor(faces): f = faces.detach().cpu().numpy()
        else: f = faces
        return trimesh.Trimesh(vertices=v, faces=f, process=False)
