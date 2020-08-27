import math
import json
import copy
import numpy as np

from it.testing.tester import Tester

from it_clearance.testing.deglomerator import DeglomeratorClearance
from it_clearance.testing.results import AnalyzerClearance


class TesterClearance(Tester):

    compiled_cv_begin = None
    compiled_cv_direction = None
    compiled_cv_end = None
    compiled_cv_norms = None
    num_cv = None

    def __init__(self, path, file):
        super().__init__(path, file)

    def read_json(self):
        super().read_json()

        for affordance in self.configuration_data['interactions']:
            sub_working_path = self.working_path + "/" + affordance['affordance_name']
            it_descriptor = DeglomeratorClearance(sub_working_path, affordance['affordance_name'], affordance['object_name'])

            if self.num_orientations != it_descriptor.num_orientations:
                raise RuntimeError("Mismatched configured and trained num_orientations")
            if self.num_pv != it_descriptor.sample_size:
                raise RuntimeError("Mismatched configured and trained num_pv")

            if self.num_cv is None:
                self.num_cv = it_descriptor.sample_clearance_size
                increments = self.num_cv*self.num_orientations
                index1 = 0
                index2 = increments
                self.compiled_cv_begin = np.empty((self.num_it_to_test*increments, 3))
                self.compiled_cv_direction = np.empty(self.compiled_cv_begin.shape)
                self.compiled_cv_norms = np.empty(self.compiled_cv_begin.shape[0])
            elif self.num_cv != it_descriptor.sample_clearance_size:
                raise RuntimeError("Mismatched configured and trained num_cv")

            self.compiled_cv_begin[index1:index2] = it_descriptor.cv_points
            self.compiled_cv_direction[index1:index2] = it_descriptor.cv_vectors
            self.compiled_cv_norms[index1:index2] = it_descriptor.cv_vectors_norms

            index1 += increments
            index2 += increments

        self.compiled_cv_end = self.compiled_cv_begin + self.compiled_cv_direction

    def get_analyzer_clearance(self, scene, position):
        translation = np.asarray(position) - self.last_position
        self.compiled_cv_begin += translation
        self.compiled_cv_end += translation
        (__, idx_ray,
             intersections) = scene.ray.intersects_id(
                ray_origins=self.compiled_cv_begin,
                ray_directions=self.compiled_cv_direction,
                return_locations=True,
                multiple_hits=False)

        return AnalyzerClearance(idx_ray, intersections, self.num_it_to_test, self.num_cv,
                                 self.num_orientations, self.compiled_cv_begin, self.compiled_cv_norms)




    def __str__(self):
        return json.dumps(self.configuration_data, indent=4)
