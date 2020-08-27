import os
import pandas as pd
import trimesh

from it.training.sampler import OnGivenPointCloudWeightedSampler
from it.training.ibs import IBSMesh
from it_clearance.training.sampler import PropagateFromSpherePoissonDiscSamplerClearance, \
    PropagateObjectNormalFromSpherePoissonDiscSamplerClearance, PropagateNormalObjectPoissonDiscSamplerClearance
from it_clearance.training.trainer import TrainerClearance
import it.util as util
import numpy as np
import open3d as o3d

from vedo import Plotter, Spheres, load, Points, Lines, Arrows, trimesh2vtk

from it_clearance.utils import get_vtk_plotter_cv_pv

if __name__ == '__main__':
    interactions_data = pd.read_csv("./data/interactions/interaction.csv")

    to_test = 'reaching_out_low'

    ibs_init_size_sampling = 400
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

    ################################
    # GENERATING AND SEGMENTING IBS MESH
    ################################
    tri_mesh_ibs = trimesh.load_mesh(os.path.join(directory, interaction.iloc[0]['tri_mesh_ibs']))
    o3d_cloud_src_ibs = o3d.io.read_point_cloud(os.path.join(directory, interaction.iloc[0]['o3d_cloud_sources_ibs']))
    np_cloud_env = np.asarray(o3d_cloud_src_ibs.points)

    influence_radio_ratio = 1.2

    sphere_ro, sphere_center = util.influence_sphere(tri_mesh_obj, influence_radio_ratio)
    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, sphere_center, sphere_ro)

    ################################
    # SAMPLING IBS MESH
    ################################

    # pv_sampler = PoissonDiscRandomSampler( sampler_rate_ibs_samples )
    # pv_sampler = PoissonDiscWeightedSampler( rate_ibs_samples=sampler_rate_ibs_samples,
    #                                       rate_generated_random_numbers=sampler_rate_generated_random_numbers)
    # pv_sampler = OnVerticesRandomSampler()
    # pv_sampler = OnVerticesWeightedSampler( rate_generated_random_numbers=sampler_rate_generated_random_numbers )
    # pv_sampler = OnGivenPointCloudRandomSampler( np_input_cloud = np_cloud_env )
    pv_sampler = OnGivenPointCloudWeightedSampler(np_input_cloud=np_cloud_env,
                                                  rate_generated_random_numbers=sampler_rate_generated_random_numbers)

    # cv_sampler = OnIBSPoissonDiscSamplerClearance()
    # cv_sampler = OnObjectPoissonDiscSamplerClearance()
    # cv_sampler = PropagateFromSpherePoissonDiscSamplerClearance()
    # cv_sampler = PropagateObjectNormalFromSpherePoissonDiscSamplerClearance()
    cv_sampler = PropagateNormalObjectPoissonDiscSamplerClearance()
    trainer = TrainerClearance(tri_mesh_ibs=tri_mesh_ibs_segmented, tri_mesh_env=tri_mesh_env,
                               tri_mesh_obj=tri_mesh_obj, pv_sampler=pv_sampler, cv_sampler=cv_sampler)

    # VISUALIZATION
    plot = get_vtk_plotter_cv_pv(trainer.pv_points, trainer.pv_vectors, trainer.cv_points, trainer.cv_vectors,
                                 tri_mesh_env, tri_mesh_obj, tri_mesh_ibs_segmented)

    from vedo.shapes import convexHull
    ch = convexHull(trainer.cv_points).alpha(.3)

    plot.actors.append(ch)

    plot.show()
