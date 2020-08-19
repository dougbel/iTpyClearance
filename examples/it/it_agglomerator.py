import os
import numpy as np
import pandas as pd

import open3d as o3d
import trimesh
import vtk
from scipy import linalg

from transforms3d.derivations.eulerangles import z_rotation
from transforms3d.affines import compose
from vtkplotter import trimesh2vtk, Plotter, Lines, Spheres

from it.training.sampler import OnGivenPointCloudWeightedSampler
from it_clearance.training.sampler import *
from it_clearance.training.trainer import TrainerClearance
from it_clearance.training.agglomerator import AgglomeratorClearance

if __name__ == '__main__':
    interactions_data = pd.read_csv("./data/interactions/interaction.csv")

    to_test = 'reaching_out_low'
    interaction = interactions_data[interactions_data['interaction'] == to_test]

    directory = interaction.iloc[0]['directory']

    tri_mesh_env = trimesh.load_mesh(os.path.join(directory, interaction.iloc[0]['tri_mesh_env']))
    tri_mesh_obj = trimesh.load_mesh(os.path.join(directory, interaction.iloc[0]['tri_mesh_obj']))
    tri_mesh_ibs = trimesh.load_mesh(os.path.join(directory, interaction.iloc[0]['tri_mesh_ibs']))
    o3d_cloud_src_ibs = o3d.io.read_point_cloud(os.path.join(directory, interaction.iloc[0]['o3d_cloud_sources_ibs']))
    np_cloud_env = np.asarray(o3d_cloud_src_ibs.points)

    influence_radio_ratio = 1.2
    sphere_ro, sphere_center = util.influence_sphere(tri_mesh_obj, radio_ratio=influence_radio_ratio)
    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, sphere_center, sphere_ro)

    tri_mesh_obj.visual.face_colors = [0, 255, 0, 255]
    tri_mesh_env.visual.face_colors = [100, 100, 100, 255]
    tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 255, 100]

    rate_generated_random_numbers = 500

    pv_sampler = OnGivenPointCloudWeightedSampler(np_cloud_env, rate_generated_random_numbers)

    cv_sampler = PropagateFromSpherePoissonDiscSamplerClearance()
    trainer = TrainerClearance(tri_mesh_ibs=tri_mesh_ibs_segmented, tri_mesh_env=tri_mesh_env,
                               tri_mesh_obj=tri_mesh_obj, pv_sampler=pv_sampler, cv_sampler=cv_sampler)

    agg = AgglomeratorClearance(trainer)

    angle = (2 * math.pi / agg.ORIENTATIONS)

    vtk_env = trimesh2vtk(tri_mesh_env).lighting("plastic")
    vtk_obj = trimesh2vtk(tri_mesh_obj).lighting("plastic")

    vp = Plotter(bg="white", axes=2)

    for ori in range(agg.ORIENTATIONS):
        idx_from = ori * agg.sample_size
        idx_to = idx_from + agg.sample_size
        pv_points = agg.agglomerated_pv_points[idx_from:idx_to]
        pv_vectors = agg.agglomerated_pv_vectors[idx_from:idx_to]

        idx_from = ori * agg.cv_sample_size
        idx_to = idx_from + agg.cv_sample_size
        cv_points = agg.agglomerated_cv_points[idx_from:idx_to]
        cv_vectors = agg.agglomerated_cv_vectors[idx_from:idx_to]

        provenance_vectors = Lines(pv_points, pv_points + pv_vectors, c='red', alpha=1).lighting("plastic")
        clearance_vectors = Lines(cv_points, cv_points + cv_vectors, c='yellow', alpha=1).lighting("plastic")
        cv_from = Spheres(cv_points, r=.007, c="yellow", alpha=1).lighting("plastic")

        vtk_obj.rotateZ(angle, rad=True)
        vtk_env.rotateZ(angle, rad=True)

        # VISUALIZATION
        vp.show([clearance_vectors, provenance_vectors, cv_from, vtk_env, vtk_obj])
