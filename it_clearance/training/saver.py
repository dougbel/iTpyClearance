import math
import json
import os
import numpy as np
import open3d as o3d
from it.training.saver import Saver
from it_clearance.training.agglomerator import AgglomeratorClearance


class SaverClearance(Saver):

    def __init__(self, affordance_name, env_name, obj_name, agglomerator, max_distances, ibs_calculator, tri_mesh_obj,
                 output_subdir=None):

        assert isinstance(agglomerator, AgglomeratorClearance)

        super().__init__(affordance_name, env_name, obj_name, agglomerator, max_distances, ibs_calculator, tri_mesh_obj,
                 output_subdir)

    def _save_agglomerated_it_descriptor(self, affordance_name, obj_name, agglomerator):
        file_name_pattern = os.path.join(self.directory, "UNew_" + affordance_name + "_" +
                                         obj_name + "_descriptor_" + str(agglomerator.ORIENTATIONS))
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_pv_points)
        o3d.io.write_point_cloud(file_name_pattern + "_points.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_pv_vectors)
        o3d.io.write_point_cloud(file_name_pattern + "_vectors.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_pv_vdata)
        o3d.io.write_point_cloud(file_name_pattern + "_vdata.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_normals)
        o3d.io.write_point_cloud(file_name_pattern + "_normals_env.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_cv_points)
        o3d.io.write_point_cloud(file_name_pattern + "_clearance_points.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_cv_vectors)
        o3d.io.write_point_cloud(file_name_pattern + "_clearance_vectors.pcd", pcd, write_ascii=True)


    def _save_info(self, affordance_name, env_name, obj_name, agglomerator, max_distances, ibs_calculator,
                   tri_mesh_obj):
        data = {'it_descriptor_version': 2.1,
                'affordance_name': affordance_name,
                'env_name': env_name,
                'obj_name': obj_name,
                'sample_size': agglomerator.it_trainer.sampler.SAMPLE_SIZE,
                'orientations': agglomerator.ORIENTATIONS,
                'trainer': agglomerator.it_trainer.get_info(),
                'ibs_calculator': ibs_calculator.get_info(),
                'max_distances': max_distances.get_info()}

        with open(os.path.join(self.directory, affordance_name + '_' + obj_name + '.json'), 'w') as outfile:
            json.dump(data, outfile, indent=4)

