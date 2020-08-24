import math

import numpy as np

import trimesh
import os
from vtkplotter import trimesh2vtk, Plotter, Lines, Spheres

from it_clearance.testing.deglomerator import DeglomeratorClearance

if __name__ == '__main__':
    working_path = "./data/it/descriptors/reaching_out_low"
    affordance_name = "reaching_out_low"
    object_name = "human_reaching_out_low"
    deglmrtor = DeglomeratorClearance(working_path, affordance_name, object_name)

    tri_mesh_obj = trimesh.load_mesh(deglmrtor.object_filename())

    env_filename = os.path.join(deglmrtor.working_path, deglmrtor.affordance_name + "_" + deglmrtor.object_name + "_environment.ply")
    tri_mesh_env = trimesh.load_mesh(env_filename)

    tri_mesh_obj.visual.face_colors = [0, 255, 0, 255]
    tri_mesh_env.visual.face_colors = [100, 100, 100, 255]

    vtk_obj = trimesh2vtk(tri_mesh_obj).lighting("plastic")
    vtk_env = trimesh2vtk(tri_mesh_env).lighting("plastic")

    vp = Plotter(bg="white", axes=2)

    angle = (2 * math.pi / deglmrtor.num_orientations)

    for ori in range(deglmrtor.num_orientations):
        idx_from = ori * deglmrtor.sample_size
        idx_to = idx_from + deglmrtor.sample_size
        pv_points = deglmrtor.pv_points[idx_from:idx_to]
        pv_vectors = deglmrtor.pv_vectors[idx_from:idx_to]

        idx_from = ori * deglmrtor.sample_clearance_size
        idx_to = idx_from + deglmrtor.sample_clearance_size
        cv_points = deglmrtor.cv_points[idx_from:idx_to]
        cv_vectors = deglmrtor.cv_vectors[idx_from:idx_to]

        provenance_vectors = Lines(pv_points, pv_points + pv_vectors, c='red', alpha=1).lighting("plastic")
        clearance_vectors = Lines(cv_points, cv_points + cv_vectors, c='yellow', alpha=1).lighting("plastic")
        cv_from = Spheres(cv_points, r=.007, c="yellow", alpha=1).lighting("plastic")

        vtk_obj.rotateZ(angle*ori, rad=True)
        vtk_env.rotateZ(angle*ori, rad=True)

        # VISUALIZATION
        vp.show([clearance_vectors, provenance_vectors, cv_from, vtk_env, vtk_obj])

        vtk_obj.rotateZ(-angle * ori, rad=True)
        vtk_env.rotateZ(-angle * ori, rad=True)