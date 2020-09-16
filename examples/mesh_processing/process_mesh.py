from si.scannet.datascannet import DataScanNet
import trimesh
import gc

import numpy as np

from sklearn.preprocessing import normalize
from vedo import load, Plotter, Spheres, Lines

if __name__ == '__main__':

    env_file = "./data/it/scene0000_00_vh_clean.ply"

    ori_tri_mesh_env = trimesh.load_mesh(env_file)
    ori_tri_mesh_env.visual.face_colors = [100, 100, 100, 255]

    if ori_tri_mesh_env.is_watertight:
        print("Watertight %s", env_file)

    spheres_radio = 0.01
    sphere_diameter = spheres_radio*2

    inverted_vertex_normals = -ori_tri_mesh_env.vertex_normals
    vertices = ori_tri_mesh_env.vertices

    (index_triangle, index_ray, ray_collision_on_object) = ori_tri_mesh_env.ray.intersects_id(
        ray_origins=vertices, ray_directions=inverted_vertex_normals,
        return_locations=True, multiple_hits=False)

    ray_intersected_vects = ray_collision_on_object - vertices[index_ray]
    ray_intersected_norms = np.linalg.norm(ray_intersected_vects, axis=1)

    index_ray_bigger_than_sphere_radio = index_ray[ray_intersected_norms > sphere_diameter]

    destino = ray_collision_on_object[ray_intersected_norms > sphere_diameter]

    ray_intersected_than_sphere_radio_orig = vertices[index_ray_bigger_than_sphere_radio]
    ray_intersected_than_sphere_radio_vects = destino - ray_intersected_than_sphere_radio_orig
    ray_intersected_than_sphere_radio_norms = np.linalg.norm(ray_intersected_than_sphere_radio_vects, axis=1)
    ray_intersected_than_sphere_radio_vects_normalized = normalize(ray_intersected_than_sphere_radio_vects)

    sphere_centres = ray_intersected_than_sphere_radio_orig + ray_intersected_than_sphere_radio_vects_normalized * spheres_radio

    scene = load(env_file).c("gray")
    sph = Spheres(sphere_centres, r=.01, c="blue", alpha=.9).lighting("plastic")
    lines = Lines(ray_intersected_than_sphere_radio_orig, sphere_centres, c="blue")

    vp = Plotter(bg="white")
    vp.add(sph)
    vp.add(scene)
    vp.show()

    index_ray_no_collided = [i for i in range(vertices.shape[0]) if i not in index_ray]



    tri_mesh_env = trimesh.load_mesh(env_file)
    tri_mesh_env.visual.face_colors = [255, 10, 10, 200]
    tri_mesh_env.fill_holes()

    if tri_mesh_env.is_watertight:
        print("Watertight %s", env_file)

    scene = trimesh.Scene([tri_mesh_env, tri_mesh_env])
    scene.show()
