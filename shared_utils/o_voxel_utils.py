import torch
import numpy as np
import trimesh
from skimage import measure

def flexible_dual_grid_to_mesh(*args, **kwargs):
    """
    Reconstructs a mesh from the sparse dual grid output using Dual Contouring logic.
    This natively interprets v_feats as vertex positions and i_feats as active edges.
    """
    if len(args) < 3:
        print(f"[Comfy3D] Error: flexible_dual_grid_to_mesh received only {len(args)} args, expected 3 or 4.")
        return torch.zeros((0, 3)), torch.zeros((0, 3)).long()
        
    coords = args[0]
    v_feats = args[1]
    i_feats = args[2]
    q_feats = args[3] if len(args) > 3 else None
    
    device = coords.device
    res = kwargs.get('grid_size', 128)
    
    try:
        # Move to CPU for processing
        c = coords.detach().cpu().long().numpy()
        v = v_feats.detach().cpu().float().numpy()
        i_f = i_feats.detach().cpu().float().numpy()
        
        # 1. Compute global vertex positions
        vertices = (c + v) / res - 0.5
        
        # 2. Map coordinates to indices
        offset_c = c + 1
        grid_size = np.max(offset_c) + 2
        
        # 1D keys
        keys = offset_c[:, 0].astype(np.int64) * (grid_size**2) + \
               offset_c[:, 1].astype(np.int64) * grid_size + \
               offset_c[:, 2].astype(np.int64)
               
        voxel_indices = np.full(grid_size**3, -1, dtype=np.int32)
        voxel_indices[keys] = np.arange(len(c))
        
        faces = []
        
        def get_indices(cx, cy, cz):
            k = cx.astype(np.int64) * (grid_size**2) + cy.astype(np.int64) * grid_size + cz.astype(np.int64)
            return voxel_indices[k]

        edge_thresh = 0.5
        
        # --- X Edges ---
        active_x = i_f[:, 0] > edge_thresh
        if np.any(active_x):
            cx = offset_c[active_x]
            v1 = get_indices(cx[:, 0], cx[:, 1], cx[:, 2])
            v2 = get_indices(cx[:, 0], cx[:, 1]-1, cx[:, 2])
            v3 = get_indices(cx[:, 0], cx[:, 1]-1, cx[:, 2]-1)
            v4 = get_indices(cx[:, 0], cx[:, 1], cx[:, 2]-1)
            valid = (v1 != -1) & (v2 != -1) & (v3 != -1) & (v4 != -1)
            if np.any(valid):
                faces.append(np.stack([v1[valid], v2[valid], v3[valid]], axis=1))
                faces.append(np.stack([v1[valid], v3[valid], v4[valid]], axis=1))

        # --- Y Edges ---
        active_y = i_f[:, 1] > edge_thresh
        if np.any(active_y):
            cy = offset_c[active_y]
            v1 = get_indices(cy[:, 0], cy[:, 1], cy[:, 2])
            v2 = get_indices(cy[:, 0], cy[:, 1], cy[:, 2]-1)
            v3 = get_indices(cy[:, 0]-1, cy[:, 1], cy[:, 2]-1)
            v4 = get_indices(cy[:, 0]-1, cy[:, 1], cy[:, 2])
            valid = (v1 != -1) & (v2 != -1) & (v3 != -1) & (v4 != -1)
            if np.any(valid):
                faces.append(np.stack([v1[valid], v2[valid], v3[valid]], axis=1))
                faces.append(np.stack([v1[valid], v3[valid], v4[valid]], axis=1))

        # --- Z Edges ---
        active_z = i_f[:, 2] > edge_thresh
        if np.any(active_z):
            cz = offset_c[active_z]
            v1 = get_indices(cz[:, 0], cz[:, 1], cz[:, 2])
            v2 = get_indices(cz[:, 0]-1, cz[:, 1], cz[:, 2])
            v3 = get_indices(cz[:, 0]-1, cz[:, 1]-1, cz[:, 2])
            v4 = get_indices(cz[:, 0], cz[:, 1]-1, cz[:, 2])
            valid = (v1 != -1) & (v2 != -1) & (v3 != -1) & (v4 != -1)
            if np.any(valid):
                faces.append(np.stack([v1[valid], v2[valid], v3[valid]], axis=1))
                faces.append(np.stack([v1[valid], v3[valid], v4[valid]], axis=1))
            
        if not faces:
            return torch.zeros((0, 3), device=device), torch.zeros((0, 3), device=device).long()
            
        faces_array = np.concatenate(faces, axis=0)
        return torch.from_numpy(vertices).float().to(device), \
               torch.from_numpy(faces_array).long().to(device)

    except Exception as e:
        print(f"[Comfy3D] Vectorized FDG shim error: {e}")
        import traceback
        traceback.print_exc()
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
            print("[Comfy3D] to_glb error: Received empty vertex array")
            return trimesh.Trimesh(vertices=np.zeros((0,3)), faces=np.zeros((0,3)))

        base_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        print(f"[Comfy3D] to_glb: Base mesh created ({len(v)} verts, {len(f)} faces)")
        
        decimation_target = kwargs.get('decimation_target', 200000)
        remesh = kwargs.get('remesh', True)
        print(f"[Comfy3D] to_glb: Simplifying to {decimation_target} faces...")
        simplifier = Trellis2Simplify()
        simplified_mesh = simplifier.simplify(base_mesh, target_face_count=decimation_target, remesh=remesh)[0]
        print(f"[Comfy3D] to_glb: Simplification complete ({len(simplified_mesh.vertices)} verts)")
        
        print("[Comfy3D] to_glb: Unwrapping UVs...")
        unwrapper = Trellis2UVUnwrap()
        unwrapped_mesh = unwrapper.unwrap(simplified_mesh)[0]
        
        print("[Comfy3D] to_glb: Rasterizing PBR textures...")
        voxelgrid = {
            'coords': coords, 'attrs': attr_volume, 'layout': attr_layout,
            'voxel_size': kwargs.get('voxel_size', 1.0),
            'original_vertices': v, 'original_faces': f
        }
        rasterizer = Trellis2RasterizePBR()
        textured_mesh = rasterizer.rasterize(unwrapped_mesh, voxelgrid, texture_size=kwargs.get('texture_size', 1024))[0]
        print("[Comfy3D] to_glb: Texture baking successful!")
        return textured_mesh
        
    except Exception as e:
        print(f"[Comfy3D] to_glb failure: {e}")
        if torch.is_tensor(vertices): v = vertices.detach().cpu().numpy()
        else: v = vertices
        if torch.is_tensor(faces): f = faces.detach().cpu().numpy()
        else: f = faces
        return trimesh.Trimesh(vertices=v, faces=f, process=False)
