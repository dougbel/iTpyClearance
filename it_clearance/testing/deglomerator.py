import json
from os import path
import numpy as np
import open3d as o3d

from it.testing.deglomerator import Deglomerator


class DeglomeratorClearance(Deglomerator):
    cv_points = None
    cv_vectors = None
    cv_vectors_norms = None
    sample_clearance_size = None

    def __init__(self, working_path, affordance_name, object_name):
        super().__init__(working_path, affordance_name, object_name)

    def read_definition(self):
        super().read_definition()
        self.sample_clearance_size = self.definition["trainer"]["cv_sampler"]["sample_clearance_size"]

    def readAgglomeratedDescriptor(self):
        super().readAgglomeratedDescriptor()
        base_nameU = self.get_agglomerated_files_name_pattern()
        self.cv_points = np.asarray(o3d.io.read_point_cloud(base_nameU + "_clearance_points.pcd").points)
        self.cv_vectors = np.asarray(o3d.io.read_point_cloud(base_nameU + "_clearance_vectors.pcd").points)
        self.cv_vectors_norms = np.asarray(o3d.io.read_point_cloud(base_nameU + "_clearance_vdata.pcd").points)[:, 0]
