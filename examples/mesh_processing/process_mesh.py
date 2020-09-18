
import trimesh
import gc

import numpy as np

from sklearn.preprocessing import normalize
from trimesh.sample import sample_surface_even
from vedo import load, Plotter, Spheres, vtk2trimesh
from it.util import sample_points_poisson_disk_radius, sample_points_poisson_disk, get_normal_nearest_point_in_mesh

if __name__ == '__main__':

    env_file = "./data/it/small_scene0000_00_vh_clean.ply"
    vedo_env = load(env_file).c("gray").bc("t")#.fillHoles(1)
    ori_tri_mesh_env = vtk2trimesh(vedo_env)

    if ori_tri_mesh_env.is_watertight:
        print("Watertight %s", env_file)

    spheres_radio = 0.01
    sphere_diameter = spheres_radio*2

    # #### using vertices of mesh
    # seed_inverted_normals = -ori_tri_mesh_env.vertex_normals
    # seed_points = ori_tri_mesh_env.vertices

    # seed_points, seed_normals = sample_points_poisson_disk_radius(ori_tri_mesh_env, radius=free_ray_spheres_radio/2)
    # seed_inverted_normals = -seed_normals

    precursor_points, __ = sample_points_poisson_disk_radius(ori_tri_mesh_env, radius=spheres_radio/2)
    seed_points = sample_points_poisson_disk(ori_tri_mesh_env, precursor_points.shape[0])
    seed_normals = get_normal_nearest_point_in_mesh(ori_tri_mesh_env, seed_points)
    seed_inverted_normals = -seed_normals

    # seed_points, face_index = sample_surface_even(ori_tri_mesh_env, ori_tri_mesh_env.vertices.shape[0], radius=free_ray_spheres_radio/2)
    # seed_inverted_normals = -ori_tri_mesh_env.face_normals[face_index]

    (index_triangle, index_ray, ray_collision_on_object) = ori_tri_mesh_env.ray.intersects_id(
        ray_origins=seed_points, ray_directions=seed_inverted_normals,
        return_locations=True, multiple_hits=False)

    # #### collided rays
    ray_intersected_vects = ray_collision_on_object - seed_points[index_ray]
    ray_intersected_norms = np.linalg.norm(ray_intersected_vects, axis=1)

    index_ray_bigger_than_sphere_radio = index_ray[ray_intersected_norms > sphere_diameter]

    destino = ray_collision_on_object[ray_intersected_norms > sphere_diameter]

    ray_intersected_than_sphere_radio_orig = seed_points[index_ray_bigger_than_sphere_radio]
    ray_intersected_than_sphere_radio_vects = destino - ray_intersected_than_sphere_radio_orig
    ray_intersected_than_sphere_radio_norms = np.linalg.norm(ray_intersected_than_sphere_radio_vects, axis=1)
    ray_intersected_than_sphere_radio_vects_normalized = normalize(ray_intersected_than_sphere_radio_vects)

    sphere_centres = ray_intersected_than_sphere_radio_orig + ray_intersected_than_sphere_radio_vects_normalized * spheres_radio

    sph_ray_collided = Spheres(sphere_centres, r=spheres_radio, c="blue", alpha=.9, res=2).lighting("plastic")
    # lines = Lines(ray_intersected_than_sphere_radio_orig, sphere_centres, c="blue")

    sph_ray_collided.cutWithMesh(vedo_env)


    # ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
    # ####  Section with rays no collided

    free_ray_spheres_radio = 0.05

    # seed_points, seed_normals = sample_points_poisson_disk_radius(ori_tri_mesh_env, radius=free_ray_spheres_radio/2)
    # seed_inverted_normals = -seed_normals

    precursor_points, __ = sample_points_poisson_disk_radius(ori_tri_mesh_env, radius=free_ray_spheres_radio/2)
    seed_points = sample_points_poisson_disk(ori_tri_mesh_env, precursor_points.shape[0])
    seed_normals = get_normal_nearest_point_in_mesh(ori_tri_mesh_env, seed_points)
    seed_inverted_normals = -seed_normals

    # seed_points, face_index = sample_surface_even(ori_tri_mesh_env, ori_tri_mesh_env.vertices.shape[0], radius=free_ray_spheres_radio/2)
    # seed_inverted_normals = -ori_tri_mesh_env.face_normals[face_index]

    (index_triangle, index_ray, ray_collision_on_object) = ori_tri_mesh_env.ray.intersects_id(
        ray_origins=seed_points, ray_directions=seed_inverted_normals,
        return_locations=True, multiple_hits=False)


    index_ray_no_collided = [i for i in range(seed_points.shape[0]) if i not in index_ray]
    ray_no_collided_vects_normalized = normalize(seed_inverted_normals[index_ray_no_collided])
    ray_no_collided_vects_orig = seed_points[index_ray_no_collided]

    sphere_centres = ray_no_collided_vects_orig + ray_no_collided_vects_normalized * free_ray_spheres_radio

    sph_ray_no_collided = Spheres(sphere_centres, r=free_ray_spheres_radio, c="orange", alpha=.9, res=2).lighting("plastic")

    sph_ray_no_collided.cutWithMesh(vedo_env)



    vp = Plotter(bg="white")
    vp.add(sph_ray_no_collided)
    vp.add(vedo_env)
    vp.add(sph_ray_collided)
    vp.show()

    print("he")