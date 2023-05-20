# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
 
import contextlib
import pathlib
import warnings
from typing import IO, ContextManager, Optional, Union
 
import numpy as np
import torch
from iopath.common.file_io import PathManager
from PIL import Image

 
@contextlib.contextmanager
def nullcontext(x):
    """
    This is just like contextlib.nullcontext but also works in Python 3.6.
    """
    yield x

 
PathOrStr = Union[pathlib.Path, str]

 
def _open_file(f, path_manager: PathManager, mode: str = "r") -> ContextManager[IO]:
    if isinstance(f, str):
        f = path_manager.open(f, mode)
        return contextlib.closing(f)
    elif isinstance(f, pathlib.Path):
        f = f.open(mode)
        return contextlib.closing(f)
    else:
        return nullcontext(f)

 
def _make_tensor(
    data, cols: int, dtype: torch.dtype, device= "cpu"
) -> torch.Tensor:
    """
    Return a 2D tensor with the specified cols and dtype filled with data,
    even when data is empty.
    """
    if not len(data):
        return torch.zeros((0, cols), dtype=dtype, device=device)
 
    return torch.tensor(data, dtype=dtype, device=device)

 
def _check_faces_indices(
    faces_indices: torch.Tensor, max_index: int, pad_value: Optional[int] = None
) -> torch.Tensor:
    if pad_value is None:
        mask = torch.ones(faces_indices.shape[:-1]).bool()  # Keep all faces
    else:
        # pyre-fixme[16]: `torch.ByteTensor` has no attribute `any`
        mask = faces_indices.ne(pad_value).any(dim=-1)
    if torch.any(faces_indices[mask] >= max_index) or torch.any(
        faces_indices[mask] < 0
    ):
        warnings.warn("Faces have invalid indices")
    return faces_indices

 
def _read_image(file_name: str, path_manager: PathManager, format=None):
    """
    Read an image from a file using Pillow.
    Args:
        file_name: image file path.
        path_manager: PathManager for interpreting file_name.
        format: one of ["RGB", "BGR"]
    Returns:
        image: an image of shape (H, W, C).
    """
    if format not in ["RGB", "BGR"]:
        raise ValueError("format can only be one of [RGB, BGR]; got %s", format)
    with path_manager.open(file_name, "rb") as f:
        # pyre-fixme[6]: Expected `Union[str, typing.BinaryIO]` for 1st param but
        #  got `Union[typing.IO[bytes], typing.IO[str]]`.
        image = Image.open(f)
        if format is not None:
            # PIL only supports RGB. First convert to RGB and flip channels
            # below for BGR.
            image = image.convert("RGB")
        image = np.asarray(image).astype(np.float32)
        if format == "BGR":
            image = image[:, :, ::-1]
        return image
    
from glob import glob
import numpy as np
import trimesh
import os
from copy import deepcopy
import pybullet as pb
import data.objects as objects
from data_making import extract_urdf
import torch
from typing import List, Optional
from typing import Union
Device = Union[str, torch.device]
from iopath.common.file_io import PathManager
from pathlib import Path
import contextlib
# from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
# from pytorch3d.ops.sample_points_from_meshes import _rand_barycentric_coords
# from pytorch3d.loss import chamfer_distance as cuda_cd
# from pytorch3d.io.obj_io import load_obj

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def urdf_to_mesh(filepath, dataset):
    """
    Receives path to object index containing the .URDF files, it extracts verts and faces, and returns the corresponding mesh.

    If dataset=='PartNetMobility', the path directory tree should be as follows:
    - objects
    |   - Object ID  <-- filepath
    |   |   - textured_objs
    |   |   |   - ...obj
    |   |- ...

    If dataset=='ShapeNetCore', the path directory tree should be as follows:
    - ShapeNetCoreV2  
    |   - Category ID
    |   |   - Object ID   <-- filepath
    |   |   |   - model.urdf 
    |   |   |   - ...
    """
    if dataset == 'ShapeNetCore':
        obj_file = os.path.join(filepath, 'models/model_normalized.obj')

        mesh = _as_mesh(trimesh.load(obj_file))
        # Convert verts and faces to np.float32
        mesh = trimesh.Trimesh(np.array(mesh.vertices).astype(np.float32), np.array(mesh.faces).astype(np.float32))

    elif dataset == 'PartNetMobility':
        total_objs = glob(os.path.join(filepath, 'textured_objs/*.obj'))
        verts = np.array([]).reshape((0,3))
        faces = np.array([]).reshape((0,3))

        mesh_list = []
        for obj_file in total_objs:
            mesh = _as_mesh(trimesh.load(obj_file))
            mesh_list.append(mesh)           
                    
        verts_list = [mesh.vertices for mesh in mesh_list]
        faces_list = [mesh.faces for mesh in mesh_list]
        faces_offset = np.cumsum([v.shape[0] for v in verts_list], dtype=np.float32)   # num of faces per mesh
        faces_offset = np.insert(faces_offset, 0, 0)[:-1]            # compute offset for faces, otherwise they all start from 0
        verts = np.vstack(verts_list)
        faces = np.vstack([face + offset for face, offset in zip(faces_list, faces_offset)])
        mesh = trimesh.Trimesh(verts, faces)

    else: 
        print("Please select a valid dataset: 'ShapeNetCore' or 'PartNetMobility'")

    return mesh


def _as_mesh(scene_or_mesh):
    # Utils function to get a mesh from a trimesh.Trimesh() or trimesh.scene.Scene()
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh


def mesh_to_pointcloud(mesh, n_samples):
    """
    This method samples n points on a mesh. The number of samples for each face is weighted by its size. 

    Params:
        mesh = trimesh.Trimesh()
        n_samples: number of total samples
    
    Returns:
        pointcloud
    """
    pointcloud, _ = trimesh.sample.sample_surface(mesh, n_samples)
    pointcloud = np.array(pointcloud).astype(np.float32)
    return pointcloud


def rotate_vertices(vertices, rot=[np.pi / 2, 0, 0]):
    """Rotate vertices by 90 deg around the x-axis. """
    new_verts = deepcopy(vertices)
    # Rotate object
    rot_Q_obj = pb.getQuaternionFromEuler(rot)
    rot_M_obj = np.array(pb.getMatrixFromQuaternion(rot_Q_obj)).reshape(3, 3)
    new_verts = np.einsum('ij,kj->ik', rot_M_obj, new_verts).transpose(1, 0)
    return new_verts


def calculate_initial_z(obj_index, scale, dataset):
    """
    Compute the mesh geometry and return the initial z-axis. This is to avoid that the object
    goes partially throught the ground.
    """
    filepath_obj = os.path.join(os.path.dirname(objects.__file__), obj_index)
    mesh = urdf_to_mesh(filepath_obj, dataset)
    verts = mesh.vertices
    pointcloud_s = scale_pointcloud(np.array(verts), scale)
    pointcloud_s_r = rotate_pointcloud(pointcloud_s)
    z_values = pointcloud_s_r[:, 2]
    height = (np.amax(z_values) - np.amin(z_values))
    return height/2


def scale_pointcloud(pointcloud, scale=0.1):
    obj = deepcopy(pointcloud)
    obj = obj * scale
    return obj


def rotate_pointcloud(pointcloud_A, rpy_BA=[np.pi / 2, 0, 0]):
    """
    The default rotation reflects the rotation used for the object during data collection.
    This calculates P_b, where P_b = R_b/a * P_a.
    R_b/a is rotation matrix of a wrt b frame.
    """
    # Rotate object
    rot_Q = pb.getQuaternionFromEuler(rpy_BA)
    rot_M = np.array(pb.getMatrixFromQuaternion(rot_Q)).reshape(3, 3)
    pointcloud_B = np.einsum('ij,kj->ki', rot_M, pointcloud_A)

    return pointcloud_B


def rotate_pointcloud_inverse(pointcloud_A, rpy_AB):
    """
    This calculates P_b, where P_b = (R_a/b)^-1 * P_a.
    R_b/a is rotation matrix of a wrt b frame."""
    rot_Q = pb.getQuaternionFromEuler(rpy_AB)
    rot_M = np.array(pb.getMatrixFromQuaternion(rot_Q)).reshape(3, 3)
    rot_M_inv = np.linalg.inv(rot_M)
    pointcloud_B = rot_M_inv @ pointcloud_A.transpose(1,0)
    pointcloud_B = pointcloud_B.transpose(1,0)

    return pointcloud_B
    

def get_ratio_urdf_deepsdf(mesh_urdf):
    """Get the ratio between the mesh in the URDF file and the processed DeepSDF mesh."""
    vertices = mesh_urdf.vertices - mesh_urdf.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    max_distances = np.max(distances)  # this is the ratio as well

    return max_distances


def preprocess_urdf():
    """The URDF mesh is processed by the loadURDF method in pybullet. It is scaled and rotated.
    This function achieves the same purpose: given a scale and a rotation matrix or quaternion, 
    it returns the vertices of the rotated and scaled mesh."""
    pass


def debug_draw_vertices_on_pb(vertices, color=[235, 52, 52], size=1):
    color = np.array(color)/255
    color_From_array = np.full(shape=vertices.shape, fill_value=color)
    pb.addUserDebugPoints(
        pointPositions=vertices,
        pointColorsRGB=color_From_array,
        pointSize=size
    )


def translate_rotate_mesh(pos_wrld_list, rot_M_wrld_list, pointclouds_list, obj_initial_pos):
    """
    Given a pointcloud (workframe), the position of the TCP (worldframe), the rotation matrix (worldframe), it returns the pointcloud in worldframe. It assumes a known position of the object.

    Params:
        pos_wrld_list: (m, 3)
        rot_M_wrld_list: (m, 3, 3)
        pointclouds_list: pointcloud in workframe (m, number_points, 3) or (number_points, 3)
        obj_initial_pos: (3,)

    Returns:
        pointcloud_wrld: (m, number_points, 3)
    """
    a = rot_M_wrld_list @ pointclouds_list.transpose(0,2,1)
    b = a.transpose(0,2,1)
    c = pos_wrld_list[:, np.newaxis, :] + b
    pointcloud_wrld = c - obj_initial_pos
    return pointcloud_wrld


# adapted from: https://github.com/facebookresearch/Active-3D-Vision-and-Touch/blob/main/pterotactyl/utility/utils.py
# returns the chamfer distance between a mesh and a point cloud (Ed. Smith)
def chamfer_distance(verts, faces, gt_points, num=1000, repeat=1):
    pred_points= batch_sample(verts, faces, num=num)
    cd, _ = cuda_cd(pred_points, gt_points, batch_reduction=None)
    if repeat > 1:
        cds = [cd]
        for i in range(repeat - 1):
            pred_points = batch_sample(verts, faces, num=num)
            cd, _ = cuda_cd(pred_points, gt_points, batch_reduction=None)
            cds.append(cd)
        cds = torch.stack(cds)
        cd = cds.mean(dim=0)
    return cd


# implemented from: https://github.com/facebookresearch/Active-3D-Vision-and-Touch/blob/main/pterotactyl/utility/utils.py, MIT License
 # sample points from a batch of meshes
def batch_sample(verts, faces, num=10000):
    # Pytorch3D based code
    bs = verts.shape[0]
    face_dim = faces.shape[0]
    vert_dim = verts.shape[1]
    # following pytorch3D convention shift faces to correctly index flatten vertices
    F = faces.unsqueeze(0).repeat(bs, 1, 1)
    F += vert_dim * torch.arange(0, bs).unsqueeze(-1).unsqueeze(-1).to(F.device)
    # flatten vertices and faces
    F = F.reshape(-1, 3)
    V = verts.reshape(-1, 3)
    with torch.no_grad():
        areas, _ = mesh_face_areas_normals(V, F)
        Ar = areas.reshape(bs, -1)
        Ar[Ar != Ar] = 0
        Ar = torch.abs(Ar / Ar.sum(1).unsqueeze(1))
        Ar[Ar != Ar] = 1

        sample_face_idxs = Ar.multinomial(num, replacement=True)
        sample_face_idxs += face_dim * torch.arange(0, bs).unsqueeze(-1).to(Ar.device)

    # Get the vertex coordinates of the sampled faces.
    face_verts = V[F]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Randomly generate barycentric coords.
    w0, w1, w2 = _rand_barycentric_coords(bs, num, V.dtype, V.device)

    # Use the barycentric coords to get a point on each sampled face.
    A = v0[sample_face_idxs]  # (N, num_samples, 3)
    B = v1[sample_face_idxs]
    C = v2[sample_face_idxs]
    samples = w0[:, :, None] * A + w1[:, :, None] * B + w2[:, :, None] * C

    return samples


# implemented from: https://github.com/facebookresearch/Active-3D-Vision-and-Touch/blob/main/pterotactyl/utility/utils.py, MIT License
# loads the initial mesh and returns vertex, and face information
def load_mesh_touch(obj):
    obj_info = load_obj(obj)
    verts = obj_info[0]
    faces = obj_info[1].verts_idx
    verts = torch.FloatTensor(verts).to(device)
    faces = torch.LongTensor(faces).to(device)
    return verts, faces

def load_obj(
    f,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    device= "cpu",
    path_manager: Optional[PathManager] = None,
):
    """
    Load a mesh from a .obj file and optionally textures from a .mtl file.
    Currently this handles verts, faces, vertex texture uv coordinates, normals,
    texture images and material reflectivity values.
 
    Note .obj files are 1-indexed. The tensors returned from this function
    are 0-indexed. OBJ spec reference: http://www.martinreddy.net/gfx/3d/OBJ.spec
 
    Example .obj file format:
    ::
        # this is a comment
        v 1.000000 -1.000000 -1.000000
        v 1.000000 -1.000000 1.000000
        v -1.000000 -1.000000 1.000000
        v -1.000000 -1.000000 -1.000000
        v 1.000000 1.000000 -1.000000
        vt 0.748573 0.750412
        vt 0.749279 0.501284
        vt 0.999110 0.501077
        vt 0.999455 0.750380
        vn 0.000000 0.000000 -1.000000
        vn -1.000000 -0.000000 -0.000000
        vn -0.000000 -0.000000 1.000000
        f 5/2/1 1/2/1 4/3/1
        f 5/1/1 4/3/1 2/4/1
 
    The first character of the line denotes the type of input:
    ::
        - v is a vertex
        - vt is the texture coordinate of one vertex
        - vn is the normal of one vertex
        - f is a face
 
    Faces are interpreted as follows:
    ::
        5/2/1 describes the first vertex of the first triangle
        - 5: index of vertex [1.000000 1.000000 -1.000000]
        - 2: index of texture coordinate [0.749279 0.501284]
        - 1: index of normal [0.000000 0.000000 -1.000000]
 
    If there are faces with more than 3 vertices
    they are subdivided into triangles. Polygonal faces are assumed to have
    vertices ordered counter-clockwise so the (right-handed) normal points
    out of the screen e.g. a proper rectangular face would be specified like this:
    ::
        0_________1
        |         |
        |         |
        3 ________2
 
    The face would be split into two triangles: (0, 2, 1) and (0, 3, 2),
    both of which are also oriented counter-clockwise and have normals
    pointing out of the screen.
 
    Args:
        f: A file-like object (with methods read, readline, tell, and seek),
           a pathlib path or a string containing a file name.
        load_textures: Boolean indicating whether material files are loaded
        create_texture_atlas: Bool, If True a per face texture map is created and
            a tensor `texture_atlas` is also returned in `aux`.
        texture_atlas_size: Int specifying the resolution of the texture map per face
            when `create_texture_atlas=True`. A (texture_size, texture_size, 3)
            map is created per face.
        texture_wrap: string, one of ["repeat", "clamp"]. This applies when computing
            the texture atlas.
            If `texture_mode="repeat"`, for uv values outside the range [0, 1] the integer part
            is ignored and a repeating pattern is formed.
            If `texture_mode="clamp"` the values are clamped to the range [0, 1].
            If None, then there is no transformation of the texture values.
        device: Device (as str or torch.device) on which to return the new tensors.
        path_manager: optionally a PathManager object to interpret paths.
 
    Returns:
        6-element tuple containing
 
        - **verts**: FloatTensor of shape (V, 3).
        - **faces**: NamedTuple with fields:
            - verts_idx: LongTensor of vertex indices, shape (F, 3).
            - normals_idx: (optional) LongTensor of normal indices, shape (F, 3).
            - textures_idx: (optional) LongTensor of texture indices, shape (F, 3).
              This can be used to index into verts_uvs.
            - materials_idx: (optional) List of indices indicating which
              material the texture is derived from for each face.
              If there is no material for a face, the index is -1.
              This can be used to retrieve the corresponding values
              in material_colors/texture_images after they have been
              converted to tensors or Materials/Textures data
              structures - see textures.py and materials.py for
              more info.
        - **aux**: NamedTuple with fields:
            - normals: FloatTensor of shape (N, 3)
            - verts_uvs: FloatTensor of shape (T, 2), giving the uv coordinate per
              vertex. If a vertex is shared between two faces, it can have
              a different uv value for each instance. Therefore it is
              possible that the number of verts_uvs is greater than
              num verts i.e. T > V.
              vertex.
            - material_colors: if `load_textures=True` and the material has associated
              properties this will be a dict of material names and properties of the form:
 
              .. code-block:: python
 
                  {
                      material_name_1:  {
                          "ambient_color": tensor of shape (1, 3),
                          "diffuse_color": tensor of shape (1, 3),
                          "specular_color": tensor of shape (1, 3),
                          "shininess": tensor of shape (1)
                      },
                      material_name_2: {},
                      ...
                  }
 
              If a material does not have any properties it will have an
              empty dict. If `load_textures=False`, `material_colors` will None.
 
            - texture_images: if `load_textures=True` and the material has a texture map,
              this will be a dict of the form:
 
              .. code-block:: python
 
                  {
                      material_name_1: (H, W, 3) image,
                      ...
                  }
              If `load_textures=False`, `texture_images` will None.
            - texture_atlas: if `load_textures=True` and `create_texture_atlas=True`,
              this will be a FloatTensor of the form: (F, texture_size, textures_size, 3)
              If the material does not have a texture map, then all faces
              will have a uniform white texture.  Otherwise `texture_atlas` will be
              None.
    """
    data_dir = "./"
    if isinstance(f, (str, bytes, Path)):
        data_dir = os.path.dirname(f)
    if path_manager is None:
        path_manager = PathManager()
    with _open_file(f, path_manager, "r") as f:
        return _load_obj(
            f,
            data_dir=data_dir,
            load_textures=load_textures,
            create_texture_atlas=create_texture_atlas,
            texture_atlas_size=texture_atlas_size,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
            device=device,
        )

def _open_file(f, path_manager, mode = "r"):
    if isinstance(f, str):
        f = path_manager.open(f, mode)
        return contextlib.closing(f)
    elif isinstance(f, pathlib.Path):
        f = f.open(mode)
        return contextlib.closing(f)
    else:
        return nullcontext(f)