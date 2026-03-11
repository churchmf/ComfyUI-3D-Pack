import torch
import numpy as np
import trimesh
from skimage import measure

def flexible_dual_grid_to_mesh(coords, v_feats, i_feats, q_feats, **kwargs):
    """
    Reconstructs a mesh from the sparse dual grid output using Dual Contouring logic.
    This natively interprets v_feats as vertex positions and i_feats as active edges.
    """
    device = coords.device
    res = kwargs.get('grid_size', 128)
    
    try:
        # Move to CPU for processing
        c = coords.detach().cpu().long().numpy()
        v = v_feats.detach().cpu().float().numpy()
        i_f = i_feats.detach().cpu().float().numpy()
        
        # 1. Compute global vertex positions
        # In Dual Grid, each active cell contains 1 vertex, shifted by v_feats.
        # The scale is usually: V = (c + v) / res - 0.5
        vertices = (c + v) / res - 0.5
        
        # 2. Map coordinates to indices for fast neighbor lookup
        # We offset by 1 to handle negative neighbor lookups safely
        offset_c = c + 1
        grid_size = np.max(offset_c) + 2
        
        # 1D keys for each coordinate (using int64 to prevent overflow)
        keys = offset_c[:, 0].astype(np.int64) * (grid_size**2) + \
               offset_c[:, 1].astype(np.int64) * grid_size + \
               offset_c[:, 2].astype(np.int64)
               
        voxel_indices = np.full(grid_size**3, -1, dtype=np.int32)
        voxel_indices[keys] = np.arange(len(c))
        
        faces = []
        
        # Helper to get vertex indices for given coordinate lists
        def get_indices(cx, cy, cz):
            k = cx.astype(np.int64) * (grid_size**2) + cy.astype(np.int64) * grid_size + cz.astype(np.int64)
            return voxel_indices[k]

        # Threshold for active edges
        edge_thresh = 0.5
        
        # --- X Edges ---
        active_x = i_f[:, 0] > edge_thresh
        if np.any(active_x):
            cx = offset_c[active_x]
            # 4 surrounding cells for an X edge: (0,0,0), (0,-1,0), (0,-1,-1), (0,0,-1)
            v1 = get_indices(cx[:, 0], cx[:, 1], cx[:, 2])
            v2 = get_indices(cx[:, 0], cx[:, 1]-1, cx[:, 2])
            v3 = get_indices(cx[:, 0], cx[:, 1]-1, cx[:, 2]-1)
            v4 = get_indices(cx[:, 0], cx[:, 1], cx[:, 2]-1)
            
            valid = (v1 != -1) & (v2 != -1) & (v3 != -1) & (v4 != -1)
            f1 = np.stack([v1[valid], v2[valid], v3[valid]], axis=1)
            f2 = np.stack([v1[valid], v3[valid], v4[valid]], axis=1)
            faces.extend([f1, f2])

        # --- Y Edges ---
        active_y = i_f[:, 1] > edge_thresh
        if np.any(active_y):
            cy = offset_c[active_y]
            # 4 surrounding cells for a Y edge: (0,0,0), (0,0,-1), (-1,0,-1), (-1,0,0)
            v1 = get_indices(cy[:, 0], cy[:, 1], cy[:, 2])
            v2 = get_indices(cy[:, 0], cy[:, 1], cy[:, 2]-1)
            v3 = get_indices(cy[:, 0]-1, cy[:, 1], cy[:, 2]-1)
            v4 = get_indices(cy[:, 0]-1, cy[:, 1], cy[:, 2])
            
            valid = (v1 != -1) & (v2 != -1) & (v3 != -1) & (v4 != -1)
            f1 = np.stack([v1[valid], v2[valid], v3[valid]], axis=1)
            f2 = np.stack([v1[valid], v3[valid], v4[valid]], axis=1)
            faces.extend([f1, f2])

        # --- Z Edges ---
        active_z = i_f[:, 2] > edge_thresh
        if np.any(active_z):
            cz = offset_c[active_z]
            # 4 surrounding cells for a Z edge: (0,0,0), (-1,0,0), (-1,-1,0), (0,-1,0)
            v1 = get_indices(cz[:, 0], cz[:, 1], cz[:, 2])
            v2 = get_indices(cz[:, 0]-1, cz[:, 1], cz[:, 2])
            v3 = get_indices(cz[:, 0]-1, cz[:, 1]-1, cz[:, 2])
            v4 = get_indices(cz[:, 0], cz[:, 1]-1, cz[:, 2])
            
            valid = (v1 != -1) & (v2 != -1) & (v3 != -1) & (v4 != -1)
            f1 = np.stack([v1[valid], v2[valid], v3[valid]], axis=1)
            f2 = np.stack([v1[valid], v3[valid], v4[valid]], axis=1)
            faces.extend([f1, f2])
            
        if not faces:
            # Fallback point cloud
            return torch.zeros((0, 3), device=device), torch.zeros((0, 3), device=device).long()
            
        faces_array = np.concatenate(faces, axis=0)
        
        return torch.from_numpy(vertices).float().to(device), \
               torch.from_numpy(faces_array).long().to(device)

    except Exception as e:
        print(f"[Comfy3D] Vectorized FDG shim error: {e}")
        import traceback
        traceback.print_exc()
        return torch.zeros((0, 3), device=device), torch.zeros((0, 3), device=device).long()

def to_glb(vertices, faces, attr_volume, coords, attr_layout, **kwargs):
    import trimesh
    import torch
    import sys
    
    # Try to use the modular TRELLIS2 nodes to do the heavy lifting
    try:
        print("[Comfy3D] to_glb: Starting texture baking pipeline...")
        # Fetch the classes directly from ComfyUI's global node registry to avoid import collisions
        import nodes
        Trellis2Simplify = nodes.NODE_CLASS_MAPPINGS["Trellis2Simplify"]
        Trellis2UVUnwrap = nodes.NODE_CLASS_MAPPINGS["Trellis2UVUnwrap"]
        Trellis2RasterizePBR = nodes.NODE_CLASS_MAPPINGS["Trellis2RasterizePBR"]
        
        if torch.is_tensor(vertices): v = vertices.detach().cpu().numpy()
        else: v = vertices
        if torch.is_tensor(faces): f = faces.detach().cpu().numpy()
        else: f = faces
        
        # 1. Base Mesh
        base_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        print(f"[Comfy3D] to_glb: Base mesh created ({len(v)} verts, {len(f)} faces)")
        
        # 2. Simplify
        decimation_target = kwargs.get('decimation_target', 200000)
        remesh = kwargs.get('remesh', True)
        print(f"[Comfy3D] to_glb: Simplifying to {decimation_target} faces (remesh={remesh})...")
        simplifier = Trellis2Simplify()
        simplified_mesh = simplifier.simplify(
            base_mesh, 
            target_face_count=decimation_target, 
            remesh=remesh,
            remesh_band=kwargs.get('remesh_band', 1.0)
        )[0]
        print(f"[Comfy3D] to_glb: Simplification complete ({len(simplified_mesh.vertices)} verts, {len(simplified_mesh.faces)} faces)")
        
        # 3. UV Unwrap
        print("[Comfy3D] to_glb: Unwrapping UVs (xatlas)...")
        unwrapper = Trellis2UVUnwrap()
        unwrapped_mesh = unwrapper.unwrap(simplified_mesh)[0]
        print(f"[Comfy3D] to_glb: UV Unwrap complete ({len(unwrapped_mesh.vertices)} verts)")
        
        # 4. Rasterize PBR
        print("[Comfy3D] to_glb: Rasterizing PBR textures...")
        voxelgrid = {
            'coords': coords,
            'attrs': attr_volume,
            'layout': attr_layout,
            'voxel_size': kwargs.get('voxel_size', 1.0),
            'original_vertices': v,
            'original_faces': f
        }
        rasterizer = Trellis2RasterizePBR()
        texture_size = kwargs.get('texture_size', 1024)
        textured_mesh = rasterizer.rasterize(unwrapped_mesh, voxelgrid, texture_size=texture_size)[0]
        print("[Comfy3D] to_glb: Texture baking successful!")
        
        return textured_mesh
        
    except Exception as e:
        print(f"[Comfy3D] to_glb texture baking failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to grey mesh if it fails
        if torch.is_tensor(vertices): v = vertices.detach().cpu().numpy()
        else: v = vertices
        if torch.is_tensor(faces): f = faces.detach().cpu().numpy()
        else: f = faces
        return trimesh.Trimesh(vertices=v, faces=f, process=False)

class postprocess:
    @staticmethod
    def to_glb(*args, **kwargs): return to_glb(*args, **kwargs)

class convert:
    @staticmethod
    def flexible_dual_grid_to_mesh(*args, **kwargs): return flexible_dual_grid_to_mesh(*args, **kwargs)
