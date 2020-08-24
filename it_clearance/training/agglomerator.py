import math

import numpy as np
from transforms3d.derivations.eulerangles import z_rotation

from it.training.agglomerator import Agglomerator
from it_clearance.training.trainer import TrainerClearance


class AgglomeratorClearance(Agglomerator):

    def __init__(self, it_trainer, num_orientations=8):
        assert isinstance(it_trainer, TrainerClearance)

        super().__init__(it_trainer, num_orientations)

        orientations = [x * (2 * math.pi / self.ORIENTATIONS) for x in range(0, self.ORIENTATIONS)]

        agglomerated_cv_points = []
        agglomerated_cv_vectors = []
        agglomerated_cv_vdata = []

        self.sample_clearance_size = it_trainer.cv_points.shape[0]
        cv_vdata = np.zeros((self.sample_clearance_size, 3), np.float64)
        cv_vdata[:, 0:1] = it_trainer.cv_norms.reshape(-1, 1)

        for angle in orientations:
            rotation = z_rotation(angle)
            agglomerated_cv_points.append(np.dot(it_trainer.cv_points, rotation.T))
            agglomerated_cv_vectors.append(np.dot(it_trainer.cv_vectors, rotation.T))
            agglomerated_cv_vdata.append(cv_vdata)

        self.agglomerated_cv_points = np.asarray(agglomerated_cv_points).reshape(-1, 3)
        self.agglomerated_cv_vectors = np.asarray(agglomerated_cv_vectors).reshape(-1, 3)
        self.agglomerated_cv_vdata = np.asarray(agglomerated_cv_vdata).reshape(-1, 3)
