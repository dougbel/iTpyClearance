import time
import math

import pandas as pd
import numpy as np
from tqdm import trange

from it import util
from it_clearance.testing.tester import TesterClearance


class EnviroTesterClearance(TesterClearance):

    def start_full_test(self, environment, points_to_test, np_env_normals):
        full_data_frame = pd.DataFrame(columns=['interaction',
                                                'point_x', 'point_y', 'point_z',
                                                'point_nx', 'point_ny', 'point_nz', 'diff_ns',
                                                'diff_ns_z_angle',
                                                'score', 'missings',
                                                'angle', 'orientation',
                                                'cv_collided', 'cv_collided_per'])

        # for testing_point in points_to_test:
        for i in trange(points_to_test.shape[0], desc="Testing environment"):
            testing_point = points_to_test[i]
            env_normal = np_env_normals[i]

            pv_analyzer = None
            for idx_aff in range(len(self.affordances)):
                affordance = self.affordances[idx_aff][0]  # it only get the interaction not the OBJECT's NAME
                affordance_env_normal = self.envs_normals[idx_aff]
                normals_angle = util.angle_between(affordance_env_normal, env_normal)
                z_axis = [0, 0, 1]
                z_angle_trained = util.angle_between(affordance_env_normal, z_axis)
                z_angle_env_pos = util.angle_between(env_normal, z_axis)
                diff_ns_z_angle = abs(z_angle_trained - z_angle_env_pos)

                if normals_angle > math.pi / 3 and diff_ns_z_angle > math.pi / 3:
                    score = math.nan
                    missing = math.nan
                    angle = math.nan
                    orientation = math.nan
                    cv_collided = math.nan
                    cv_collided_per = math.nan
                    full_data_frame.loc[len(full_data_frame)] = [affordance,
                                                             testing_point[0], testing_point[1], testing_point[2],
                                                             env_normal[0], env_normal[1], env_normal[2],
                                                             normals_angle,
                                                             diff_ns_z_angle,
                                                             score, missing,
                                                             angle, orientation,
                                                             cv_collided, cv_collided_per]
                else:
                    if pv_analyzer is None:
                        pv_analyzer = self.get_analyzer(environment, testing_point)
                        cv_analyzer = self.get_analyzer_clearance(environment, testing_point)
                        pv_analyzer.measure_scores()
                    for orientation in range(self.num_orientations):
                        full_data_frame.loc[len(full_data_frame)] = [affordance,
                                     testing_point[0], testing_point[1], testing_point[2],
                                     env_normal[0], env_normal[1], env_normal[2],
                                     normals_angle,
                                     diff_ns_z_angle,
                                     pv_analyzer.results.distances_summary[idx_aff][orientation],
                                     pv_analyzer.results.missed[idx_aff][orientation],
                                     (2 * math.pi / pv_analyzer.num_orientations) * orientation,
                                     orientation,
                                     cv_analyzer.results.resumed_smaller_norm_by_inter_and_ori[idx_aff][orientation],
                                     cv_analyzer.results.percentage_smaller_norm_by_inter_ori[idx_aff][orientation]]

        return full_data_frame