import math

from collections import Counter
from abc import ABC, abstractmethod
import numpy as np

import trimesh

import it.util as util


class SamplerClearance(ABC):
    SAMPLE_SIZE = 128

    # input for sampling execution
    tri_mesh_ibs = None
    tri_mesh_obj = None
    # outputs
    cv_points = np.array([])
    cv_vectors = np.array([])
    cv_norms = np.array([])

    def __init__(self):
        super().__init__()

    def execute(self, tri_mesh_ibs, tri_mesh_obj):
        """
        Generate the clearance vectors
        Parameters
        ----------
        tri_mesh_ibs: mesh structure of Interaction Bisector Surface (IBS)
        tri_mesh_obj: mesh structure of object

        Returns
        -------
        generate instance variables cv_points, cv_vectors, cv_norms associated with the clearance vectors
        """
        self.tri_mesh_ibs = tri_mesh_ibs
        self.tri_mesh_obj = tri_mesh_obj
        np_src_cloud = self.get_source_cloud()
        self.calculate_clearance_vectors(np_src_cloud)

    @abstractmethod
    def calculate_clearance_vectors(self, np_src_cloud):
        pass

    @abstractmethod
    def get_source_cloud(self):
        pass

    def get_info(self):
        info = {}
        info['sampler_clearance_name'] = self.__class__.__name__
        info['sample_clearance_size'] = self.SAMPLE_SIZE
        return info


class OnIBSPoissonDiscSamplerClearance(SamplerClearance):
    """
    Generates provenance vectors by sampling on the IBS and extend them to the nearest point in the object
    """

    def __init__(self):
        super().__init__()

    def get_source_cloud(self):
        return util.sample_points_poisson_disk(self.tri_mesh_ibs, self.SAMPLE_SIZE)

    def calculate_clearance_vectors(self, np_src_cloud):
        self.cv_points = np_src_cloud
        (closest_points_in_obj, norms, __) = self.tri_mesh_obj.nearest.on_surface(self.cv_points)
        self.cv_vectors = closest_points_in_obj - self.cv_points
        self.cv_norms = norms


class OnObjectPoissonDiscSamplerClearance(SamplerClearance):
    """
    Generates provenance vectors by
    1) poisson disc sampling on object
    2) finding the nearest point from OBJECT SAMPLES to IBS
    """
    influence_radio_ratio = 1.2

    def __init__(self):
        super().__init__()

    def get_source_cloud(self):
        return util.sample_points_poisson_disk(self.tri_mesh_obj, self.SAMPLE_SIZE)

    def calculate_clearance_vectors(self, np_src_cloud):
        sphere_ro, sphere_center = util.influence_sphere(self.tri_mesh_obj, self.influence_radio_ratio)
        sphere = trimesh.primitives.Sphere(radius=sphere_ro, center=sphere_center)

        wrapper = self.tri_mesh_ibs + sphere

        (self.cv_points, self.cv_norms, __) = wrapper.nearest.on_surface(np_src_cloud)

        self.cv_vectors = np_src_cloud - self.cv_points


class PropagateFromSpherePoissonDiscSamplerClearance(OnObjectPoissonDiscSamplerClearance):
    """
    Generates provenance vectors by
    1) sampling on a sphere of influence,
    2) generate rays from samples to the sphere centre
    3) find intersection of rays in object obtaining OBJECT SAMPLES
    4) finding nearest point from OBJECT SAMPLES to IBS
    """
    def __init__(self):
        super().__init__()

    def get_source_cloud(self):
        sphere_ro, sphere_center = util.influence_sphere(self.tri_mesh_obj, self.influence_radio_ratio)
        sphere = trimesh.primitives.Sphere(radius=sphere_ro, center=sphere_center)
        sphere_samples = util.sample_points_poisson_disk(sphere, self.SAMPLE_SIZE)

        rays_to_center = sphere_center - sphere_samples

        (__, __, np_src_cloud) = self.tri_mesh_obj.ray.intersects_id(
                                                ray_origins=sphere_samples, ray_directions=rays_to_center,
                                                return_locations=True, multiple_hits=False)

        return np_src_cloud
