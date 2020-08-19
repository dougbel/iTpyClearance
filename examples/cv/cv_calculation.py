import os
import pandas as pd

from it.training.sampler import OnGivenPointCloudWeightedSampler
from it_clearance.training.maxdistancescalculator import MaxDistancesCalculator
from it_clearance.training.ibs import IBSMesh
from it_clearance.training.sampler import *
from it_clearance.training.trainer import TrainerClearance
# from it_clearance.training.agglomerator import Agglomerator
# from it_clearance.training.saver import Saver

from vtkplotter import Plotter, Spheres, load, Points, Lines, Arrows, trimesh2vtk

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
    influence_radio_ratio = 1.2
    ibs_calculator = IBSMesh(ibs_init_size_sampling, ibs_resamplings)
    ibs_calculator.execute(tri_mesh_env_segmented, tri_mesh_obj)

    tri_mesh_ibs = ibs_calculator.get_trimesh()

    sphere_ro, sphere_center = util.influence_sphere(tri_mesh_obj, influence_radio_ratio)
    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, sphere_center, sphere_ro)

    np_cloud_env = ibs_calculator.points[: ibs_calculator.size_cloud_env]

    ################################
    # SAMPLING IBS MESH
    ################################

    # sampler = PoissonDiscRandomSampler( sampler_rate_ibs_samples )
    # sampler = PoissonDiscWeightedSampler( rate_ibs_samples=sampler_rate_ibs_samples,
    #                                       rate_generated_random_numbers=sampler_rate_generated_random_numbers)
    # sampler = OnVerticesRandomSampler()
    # sampler = OnVerticesWeightedSampler( rate_generated_random_numbers=sampler_rate_generated_random_numbers )
    # sampler = OnGivenPointCloudRandomSampler( np_input_cloud = np_cloud_env )
    sampler = OnGivenPointCloudWeightedSampler(np_input_cloud=np_cloud_env,
                                               rate_generated_random_numbers=sampler_rate_generated_random_numbers)

    # cv_sampler = OnIBSPoissonDiscSamplerClearance()
    # cv_sampler = OnObjectPoissonDiscSamplerClearance()
    cv_sampler = PropagateFromSpherePoissonDiscSamplerClearance()
    trainer = TrainerClearance(tri_mesh_ibs=tri_mesh_ibs_segmented, tri_mesh_env=tri_mesh_env,
                               tri_mesh_obj=tri_mesh_obj, pv_sampler=sampler, cv_sampler=cv_sampler)

    # VISUALIZATION
    plot = get_vtk_plotter_cv_pv(trainer.pv_points, trainer.pv_vectors, trainer.cv_points, trainer.cv_vectors,
                                 tri_mesh_env, tri_mesh_obj, tri_mesh_ibs_segmented)
    plot.show()
