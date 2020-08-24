import os

import pandas as pd

from it.training.ibs import IBSMesh
from it.training.sampler import OnGivenPointCloudWeightedSampler
from it.training.maxdistancescalculator import MaxDistancesCalculator

from it_clearance.training.sampler import *
from it_clearance.training.agglomerator import AgglomeratorClearance
from it_clearance.training.saver import SaverClearance
from it_clearance.training.trainer import TrainerClearance
from vtkplotter import Plotter, Spheres, load, Points, Lines, Arrows, trimesh2vtk

# from it_clearance.training.agglomerator import Agglomerator
# from it_clearance.training.saver import Saver
from it_clearance.utils import get_vtk_plotter_cv_pv

if __name__ == '__main__':
    interactions_data = pd.read_csv("./data/interactions/interaction.csv")

    to_test = 'reaching_out_low'
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

    ibs_init_size_sampling = 400
    ibs_resamplings = 2
    sampler_rate_ibs_samples = 5
    sampler_rate_generated_random_numbers = 500

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

    agglomerator = AgglomeratorClearance(trainer, num_orientations=8)

    max_distances = MaxDistancesCalculator(pv_points=trainer.pv_points, pv_vectors=trainer.pv_vectors,
                                            tri_mesh_obj=tri_mesh_obj, consider_collision_with_object=True,
                                            radio_ratio=influence_radio_ratio)

    output_subdir = "IBSMesh_" + str(ibs_init_size_sampling) + "_" + str(ibs_resamplings) + "_"
    output_subdir += pv_sampler.__class__.__name__ + "_" + str(sampler_rate_ibs_samples) + "_"
    output_subdir += str(sampler_rate_generated_random_numbers)+"_"
    output_subdir += cv_sampler.__class__.__name__ + "_" + str(cv_sampler.sample_size)

    SaverClearance(affordance_name, env_name, obj_name, agglomerator,
                   max_distances, ibs_calculator, tri_mesh_obj, output_subdir)

    # VISUALIZATION
    plot = get_vtk_plotter_cv_pv(trainer.pv_points, trainer.pv_vectors, trainer.cv_points, trainer.cv_vectors,
                                 tri_mesh_env, tri_mesh_obj, tri_mesh_ibs_segmented)

    plot.show()
