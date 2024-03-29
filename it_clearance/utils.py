from vedo import Lines, Spheres, trimesh2vtk, Plotter
from vedo.utils import flatten
import numpy as np

def calculate_average_distance_nearest_neighbour(points):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)
    return np.average(distances, axis=0)[1]

def central_point_mesh(tri_mesh):
    obj_min_bound = np.asarray(tri_mesh.vertices).min(axis=0)
    obj_max_bound = np.asarray(tri_mesh.vertices).max(axis=0)
    centre = np.asarray(obj_max_bound + obj_min_bound) / 2
    return centre


def get_vtk_plotter_cv_pv(pv_points, pv_vectors, cv_points, cv_vectors,
                          trimesh_env=None, trimesh_obj=None, trimesh_ibs=None):

    plot_elements = get_vtk_items_cv_pv(pv_points, pv_vectors, cv_points, cv_vectors, trimesh_env, trimesh_obj,
                                        trimesh_ibs)

    vp = Plotter(bg="white")
    vp.add(plot_elements)
    return vp


def get_vtk_items_cv_pv(pv_points, pv_vectors, cv_points, cv_vectors,
                        trimesh_env=None, trimesh_obj=None, trimesh_ibs=None):
    clearance_vectors = Lines(cv_points, cv_points + cv_vectors, c='yellow', alpha=1).lighting("plastic")
    provenance_vectors = Lines(pv_points, pv_points + pv_vectors, c='red', alpha=1).lighting("plastic")
    cv_from = Spheres(cv_points, r=.002, c="yellow", alpha=1).lighting("plastic")

    vtk_elements = [clearance_vectors, provenance_vectors, cv_from]

    if trimesh_env is not None:
        trimesh_env.visual.face_colors = [200, 200, 200, 250]
        vtk_env = trimesh2vtk(trimesh_env)
        vtk_env.lighting("plastic")
        vtk_elements.append(vtk_env)

    if trimesh_obj is not None:
        trimesh_obj.visual.face_colors = [0, 250, 0, 255]
        vtk_obj = trimesh2vtk(trimesh_obj)
        vtk_obj.lighting("plastic")
        vtk_elements.append(vtk_obj)

    if trimesh_ibs is not None:
        trimesh_ibs.visual.face_colors = [0, 0, 200, 100]
        vtk_ibs = trimesh2vtk(trimesh_ibs)
        vtk_ibs.lighting("plastic")
        vtk_elements.append(vtk_ibs)

    vtk_elements = flatten(vtk_elements)

    return vtk_elements


def get_vtk_plotter_ibs(trimesh_env, trimesh_obj, trimesh_ibs, src_cloud_env=None, src_cloud_obj=None):
    trimesh_env.visual.face_colors = [200, 200, 200, 255]
    vtk_env = trimesh2vtk(trimesh_env)
    vtk_env.lighting("plastic")

    trimesh_obj.visual.face_colors = [0, 250, 0, 255]
    vtk_obj = trimesh2vtk(trimesh_obj)
    vtk_obj.lighting("plastic")

    trimesh_ibs.visual.face_colors = [0, 0, 200, 100]
    vtk_ibs = trimesh2vtk(trimesh_ibs)
    vtk_ibs.lighting("plastic")

    plot_elements = [vtk_env, vtk_obj, vtk_ibs]

    if src_cloud_env is not None:
        vtk_src_cloud_env = Spheres(src_cloud_env, r=.007, c="blue", alpha=.9).lighting("plastic")
        plot_elements.append(vtk_src_cloud_env)
    if src_cloud_obj is not None:
        vtk_cloud_obj = Spheres(src_cloud_obj, r=.007, c="blue", alpha=.9).lighting("plastic")
        plot_elements.append(vtk_cloud_obj)

    plot_elements = flatten(plot_elements)

    vp = Plotter(bg="white")
    vp.add(plot_elements)

    return vp
