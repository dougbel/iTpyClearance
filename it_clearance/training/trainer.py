import numpy as np
import sys
from it_clearance.training.sampler import OnIBSPoissonDiscSamplerClearance
from it import util
from it.training.trainer import Trainer
import trimesh


class TrainerClearance(Trainer):
    cv_sampler = None
    cv_points = np.asarray([])
    cv_vectors = np.asarray([])
    cv_norms = np.asarray([])

    def __init__(self, tri_mesh_ibs, tri_mesh_env, tri_mesh_obj, pv_sampler, cv_sampler):
        """
        It tries to generates the clearance vectors using only the IBS surface and points related to
        Parameters
        ----------
        tri_mesh_ibs: IBS mesh
        tri_mesh_env: Environment mesh
        tri_mesh_obj: Object mesh
        sampler: sample used to select provenance vectors
        rate: The number of random
        """
        super().__init__(tri_mesh_ibs, tri_mesh_env, pv_sampler)
        self.cv_sampler = cv_sampler
        self._get_clearance_vectors(tri_mesh_ibs, tri_mesh_obj)

    def _get_clearance_vectors(self, tri_mesh_ibs, tri_mesh_obj):
        self.cv_sampler.execute(tri_mesh_ibs, tri_mesh_obj)
        self.cv_points = self.cv_sampler.cv_points
        self.cv_vectors = self.cv_sampler.cv_vectors
        self.cv_norms = self.cv_sampler.cv_norms

    def get_info(self):
        info = super().get_info()
        info['cv_sampler'] = self.cv_sampler.get_info()
        return info
