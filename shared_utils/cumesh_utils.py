import torch
import numpy as np
import trimesh
import xatlas

class CuMesh:
    def __init__(self):
        self.vertices = None
        self.faces = None
        self.trimesh_obj = None
        self.num_boundaries = 0
        self.num_boundary_loops = 0

    def init(self, vertices, faces):
        # vertices: [V, 3] tensor or numpy
        # faces: [F, 3] tensor or numpy
        self.device = vertices.device if torch.is_tensor(vertices) else 'cpu'
        
        if torch.is_tensor(vertices):
            self.vertices = vertices.detach().cpu().numpy()
        else:
            self.vertices = vertices
            
        if torch.is_tensor(faces):
            self.faces = faces.detach().cpu().numpy()
        else:
            self.faces = faces
            
        self.trimesh_obj = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)

    @property
    def num_vertices(self):
        return len(self.vertices) if self.vertices is not None else 0

    @property
    def num_faces(self):
        return len(self.faces) if self.faces is not None else 0

    def unify_face_orientations(self):
        if self.trimesh_obj:
            trimesh.repair.fix_normals(self.trimesh_obj)
            self.vertices = self.trimesh_obj.vertices
            self.faces = self.trimesh_obj.faces

    def simplify(self, target_face_count, verbose=False, **kwargs):
        if self.trimesh_obj:
            try:
                current_faces = len(self.faces)
                if current_faces > target_face_count:
                    import fast_simplification
                    ratio = target_face_count / current_faces
                    # fast_simplification expects target_reduction (how much to remove)
                    v_out, f_out = fast_simplification.simplify(self.vertices, self.faces, target_reduction=1.0 - ratio)
                    self.vertices = v_out
                    self.faces = f_out
                    self.trimesh_obj = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
                    if verbose:
                        print(f"[Comfy3D] Decimated mesh from {current_faces} to {len(self.faces)} faces")
            except Exception as e:
                print(f"[Comfy3D] Mesh simplification failed: {e}")

    def fill_holes(self, max_hole_perimeter=None):
        if self.trimesh_obj:
            trimesh.repair.fill_holes(self.trimesh_obj)
            self.vertices = self.trimesh_obj.vertices
            self.faces = self.trimesh_obj.faces

    def read(self):
        # Returns (verts, faces) as tensors on original device
        return torch.from_numpy(self.vertices.copy()).float().to(self.device), \
               torch.from_numpy(self.faces.copy()).long().to(self.device)

    def uv_unwrap(self, method='xatlas', **kwargs):
        # method is ignored for now, always use xatlas
        v = self.vertices
        f = self.faces
        atlas = xatlas.Atlas()
        atlas.add_mesh(v, f)
        atlas.generate()
        vmapping, indices, uvs = atlas[0]
        
        out_vertices = v[vmapping]
        out_faces = indices
        out_uvs = uvs
        out_vmaps = torch.from_numpy(vmapping.copy()).long().to(self.device)
        
        return torch.from_numpy(out_vertices.copy()).float().to(self.device), \
               torch.from_numpy(out_faces.copy()).long().to(self.device), \
               torch.from_numpy(out_uvs.copy()).float().to(self.device), \
               out_vmaps

    def compute_vertex_normals(self):
        if self.trimesh_obj:
            self.trimesh_obj.vertex_normals # This triggers computation

    def read_vertex_normals(self):
        if self.trimesh_obj:
            return torch.from_numpy(self.trimesh_obj.vertex_normals.copy()).float().to(self.device)
        return None

    # VAE / Reconstruction shims
    def get_edges(self): pass
    def get_boundary_info(self):
        # Returning 0 boundaries triggers early exit in fill_holes, which is safe.
        self.num_boundaries = 0
    def get_vertex_edge_adjacency(self): pass
    def get_vertex_boundary_adjacency(self): pass
    def get_manifold_boundary_adjacency(self): pass
    def read_manifold_boundary_adjacency(self): return None
    def get_boundary_connected_components(self): pass
    def get_boundary_loops(self):
        self.num_boundary_loops = 0

class cuBVH:
    def __init__(self, vertices, faces):
        """
        Functional shim for cuBVH using trimesh proximity query.
        Used for mapping PBR attributes between meshes on AMD.
        """
        if torch.is_tensor(vertices):
            v_np = vertices.detach().cpu().numpy()
        else:
            v_np = vertices
            
        if torch.is_tensor(faces):
            f_np = faces.detach().cpu().numpy()
        else:
            f_np = faces
            
        self.mesh = trimesh.Trimesh(vertices=v_np, faces=f_np, process=False)
        self.query = trimesh.proximity.ProximityQuery(self.mesh)
        self.device = vertices.device if torch.is_tensor(vertices) else 'cpu'

    def unsigned_distance(self, points, return_uvw=False):
        """
        points: [N, 3] tensor or numpy
        Returns (dist, face_id, uvw)
        Approximated using fast cKDTree for AMD performance.
        """
        from scipy.spatial import cKDTree
        import torch
        
        is_tensor = torch.is_tensor(points)
        if is_tensor:
            p_np = points.detach().cpu().numpy()
        else:
            p_np = points
            
        # 1. Fast Vertex Lookup using KDTree
        # We build the tree on vertices for speed.
        tree = cKDTree(self.mesh.vertices)
        dist, vert_indices = tree.query(p_np, workers=-1)
        
        # 2. Map Vertex IDs to Face IDs
        # To satisfy the return_uvw requirement, we need a face_id and barycentric uvw.
        # We can just pick the first face that uses the nearest vertex.
        vertex_to_face = self.mesh.vertex_faces[vert_indices]
        # vertex_faces returns -1 for padding if vertex has fewer faces. Take first valid.
        face_id = vertex_to_face[:, 0]
        
        if return_uvw:
            # We approximate by snapping to the vertex. 
            # In barycentric [w0, w1, w2], if we snap to v0, uvw is [1, 0, 0].
            # But wait, nodes_unwrap.py uses:
            #   orig_tri_verts = orig_vertices[orig_faces[face_id.long()]]
            #   valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
            # So we need uvw to match the index of vert_indices inside orig_faces[face_id].
            
            faces_of_interest = self.mesh.faces[face_id] # [N, 3]
            uvw = np.zeros((len(p_np), 3), dtype=np.float32)
            
            # For each point, find which corner of the face matches our nearest vertex
            for corner in range(3):
                mask = (faces_of_interest[:, corner] == vert_indices)
                uvw[mask, corner] = 1.0
                
            if is_tensor:
                return torch.from_numpy(dist).to(self.device), \
                       torch.from_numpy(face_id).to(self.device), \
                       torch.from_numpy(uvw).to(self.device)
            return dist, face_id, uvw
            
        if is_tensor:
            return torch.from_numpy(dist).to(self.device), \
                   torch.from_numpy(face_id).to(self.device)
        return dist, face_id

# remeshing submodule
class remeshing:
    @staticmethod
    def remesh_narrow_band_dc(v, f, *args, **kwargs):
        # For now, just return original. DC remeshing is hard to shim correctly.
        return v, f
