import math
import numpy as np

from it import util


class ResultsClearance:
    collision_vectors_norms = None
    is_smaller_norm_by_inter_and_ori = None
    resumed_smaller_norm_by_inter_and_ori = None
    percentage_smaller_norm_by_inter_ori = None

    def __init__(self, collision_vectors_norms, is_smaller_norm_by_inter_and_ori,
                 resumed_smaller_norm_by_inter_and_ori, percentage_smaller_norm_by_inter_ori):

        self.collision_vectors_norms = collision_vectors_norms
        self.is_smaller_norm_by_inter_and_ori = is_smaller_norm_by_inter_and_ori
        self.resumed_smaller_norm_by_inter_and_ori = resumed_smaller_norm_by_inter_and_ori
        self.percentage_smaller_norm_by_inter_ori = percentage_smaller_norm_by_inter_ori


class AnalyzerClearance:
    results = None

    def __init__(self, idx_ray, idx_ray_intersections, num_it_to_test, num_cv, num_orientations,
                 compiled_cv_begin, compiled_cv_norms):
        self.idx_ray = idx_ray
        self.idx_ray_intersections = idx_ray_intersections
        self.num_it_to_test = num_it_to_test
        self.num_cv_per_it_and_ori = num_cv
        self.num_orientations = num_orientations
        self.compiled_cv_begin = compiled_cv_begin
        self.compiled_cv_norms = compiled_cv_norms
        self.compare_norms_collided_and_clearance_vectors()

    def compare_norms_collided_and_clearance_vectors(self):
        if self.results is None:
            collision_vectors_norms = self._calculate_collision_vectors_norms()
            is_s_n_int_ori = self._exist_smaller_norms_compared_with_trained_by_iter_and_ori(collision_vectors_norms)
            res_sm_n_int_ori = self._count_smaller_norms_compared_with_trained(is_s_n_int_ori)
            per_sm_n_int_ori = self._calculate_percentage_smaller_norms_by_iter_and_ori(res_sm_n_int_ori)
            self.results = ResultsClearance(collision_vectors_norms, is_s_n_int_ori, res_sm_n_int_ori, per_sm_n_int_ori)
        return self.results

    def _calculate_percentage_smaller_norms_by_iter_and_ori(self, resumed_by_inter_and_ori):
        percentage_by_inter_ori = np.zeros(resumed_by_inter_and_ori.shape)
        for num_it in range(self.num_it_to_test):
            percentage_by_inter_ori[num_it, :] = resumed_by_inter_and_ori[num_it,:] / self.num_cv_per_it_and_ori
        return percentage_by_inter_ori

    def _exist_smaller_norms_compared_with_trained_by_iter_and_ori(self, collision_vectors_norms):
        comparison = collision_vectors_norms < self.compiled_cv_norms
        results_by_iter_and_ori = comparison.reshape(self.num_it_to_test,
                                                      self.num_orientations,
                                                      self.num_cv_per_it_and_ori)
        return results_by_iter_and_ori

    def _count_smaller_norms_compared_with_trained(self, results_by_iter_and_ori):
        by_cv_set = results_by_iter_and_ori.reshape(self.num_it_to_test*self.num_orientations, self.num_cv_per_it_and_ori)
        resumed_by_inter_and_ori = np.array(list(map(np.sum, by_cv_set)))
        resumed_by_inter_and_ori = resumed_by_inter_and_ori.reshape(self.num_it_to_test, self.num_orientations)

        return resumed_by_inter_and_ori

    def _calculate_collision_vectors_norms(self):
        collision_vecs_norms = np.linalg.norm(self.idx_ray_intersections - self.compiled_cv_begin[self.idx_ray], axis=1)
        all_norms = np.empty(self.compiled_cv_begin.shape[0])
        all_norms[:] = math.inf
        all_norms[self.idx_ray] = collision_vecs_norms

        return all_norms
