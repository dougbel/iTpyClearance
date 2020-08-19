import os
import pandas as pd
import trimesh

from it.training.ibs import IBSMesh
from it_clearance.utils import get_vtk_plotter_ibs
import it.util as util
import open3d as o3d

if __name__ == '__main__':
    inter_directory = "./data/interactions"
    interactions_data = pd.read_csv(os.path.join(inter_directory, "interaction.csv"))

    to_test = 'reaching_out_low'

    ibs_init_size_sampling = 2000
    ibs_resamplings = 2
    sampler_rate_ibs_samples = 5
    sampler_rate_generated_random_numbers = 500

    interaction = interactions_data[interactions_data['interaction'] == to_test]

    directory = interaction.iloc[0]['directory']

    tri_mesh_env = trimesh.load_mesh(os.path.join(directory, interaction.iloc[0]['tri_mesh_env']))
    tri_mesh_obj = trimesh.load_mesh(os.path.join(directory, interaction.iloc[0]['tri_mesh_obj']))
    obj_name = interaction.iloc[0]['obj']
    env_name = interaction.iloc[0]['env']
    affordance_name = interaction.iloc[0]['interaction']

    influence_radio_bb = 2
    extension, middle_point = util.influence_sphere(tri_mesh_obj, influence_radio_bb)
    tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(tri_mesh_env, middle_point, extension)

    # ###############################
    # GENERATING AND SEGMENTING IBS MESH
    # ###############################
    influence_radio_ratio = 1.2
    ibs_calculator = IBSMesh(ibs_init_size_sampling, ibs_resamplings)
    ibs_calculator.execute(tri_mesh_env_segmented, tri_mesh_obj)

    tri_mesh_ibs = ibs_calculator.get_trimesh()

    sphere_ro, sphere_center = util.influence_sphere(tri_mesh_obj, influence_radio_ratio)
    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, sphere_center, sphere_ro)

    np_cloud_env = ibs_calculator.points[: ibs_calculator.size_cloud_env]
    np_cloud_obj = ibs_calculator.points[ibs_calculator.size_cloud_env:]

    # ###############################
    # Saving IBS
    # ###############################
    file_name_pattern = os.path.join(inter_directory, affordance_name, affordance_name + "_" + obj_name)
    tri_mesh_ibs.export(file_name_pattern + "_ibs_mesh.ply", "ply")
    o3d_cloud_env = o3d.geometry.PointCloud()
    o3d_cloud_env.points = o3d.utility.Vector3dVector(np_cloud_env)
    o3d.io.write_point_cloud(file_name_pattern + "_o3d_cloud_sources_ibs.pcd", o3d_cloud_env, write_ascii=True)

    # VISUALIZATION
    plot = get_vtk_plotter_ibs(tri_mesh_env, tri_mesh_obj, tri_mesh_ibs_segmented, np_cloud_env, np_cloud_obj)

    plot.show()
