import math

from collections import Counter
from abc import ABC, abstractmethod
import numpy as np
import operator
import trimesh

import it.util as util


class SamplerClearance(ABC):
    sample_size = 256

    # input for sampling execution
    tri_mesh_ibs = None
    tri_mesh_obj = None
    # work variables
    np_src_cloud = np.array([])
    # outputs
    cv_points = np.array([])
    cv_vectors = np.array([])
    cv_norms = np.array([])

    def __init__(self, sample_size=None):
        super().__init__()
        if sample_size is not None:
            self.sample_size = sample_size

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
        self.get_source_cloud()
        self.calculate_clearance_vectors()

    @abstractmethod
    def calculate_clearance_vectors(self):
        pass

    @abstractmethod
    def get_source_cloud(self):
        pass

    def get_info(self):
        info = {}
        info['sampler_clearance_name'] = self.__class__.__name__
        info['sample_clearance_size'] = self.sample_size
        return info


class OnIBSPoissonDiscSamplerClearance(SamplerClearance):
    """
    Generates provenance vectors by sampling on the IBS and extend them to the nearest point in the object
    """

    def __init__(self):
        super().__init__()

    def get_source_cloud(self):
        self.np_src_cloud = util.sample_points_poisson_disk(self.tri_mesh_ibs, self.sample_size)

    def calculate_clearance_vectors(self):
        self.cv_points = self.np_src_cloud
        (closest_points_in_obj, norms, __) = self.tri_mesh_obj.nearest.on_surface(self.cv_points)
        self.cv_vectors = closest_points_in_obj - self.cv_points
        self.cv_norms = norms


class OnObjectPoissonDiscSamplerClearance(SamplerClearance):
    """
    Generates clearance vectors by
    1) poisson disc sampling on object
    2) finding the nearest point from OBJECT SAMPLES to ( IBS U Sphere_of_influence)
    """
    influence_radio_ratio = 1.2

    def get_source_cloud(self):
        self.np_src_cloud = util.sample_points_poisson_disk(self.tri_mesh_obj, self.sample_size)

    def calculate_clearance_vectors(self):
        sphere_ro, sphere_center = util.influence_sphere(self.tri_mesh_obj, self.influence_radio_ratio)
        sphere = trimesh.primitives.Sphere(radius=sphere_ro, center=sphere_center)

        wrapper = self.tri_mesh_ibs + sphere

        (self.cv_points, self.cv_norms, __) = wrapper.nearest.on_surface(self.np_src_cloud)

        self.cv_vectors = self.np_src_cloud - self.cv_points

    def get_info(self):
        info = super().get_info()
        info['influence_radio_ratio'] = self.influence_radio_ratio
        return info

class PropagateFromSpherePoissonDiscSamplerClearance(OnObjectPoissonDiscSamplerClearance):
    """
    Generates clearance vectors by
    1) sampling on a sphere of influence,
    2) generate rays from samples to the sphere centre
    3) find intersection of rays in object obtaining OBJECT SAMPLES
        IF NO INTERSECTION: find the nearest point from "circle sample" to object
    4) finding nearest point from OBJECT SAMPLES to IBS
    """


    def get_source_cloud(self):
        sphere_ro, sphere_center = util.influence_sphere(self.tri_mesh_obj, self.influence_radio_ratio)
        sphere = trimesh.primitives.Sphere(radius=sphere_ro, center=sphere_center)
        sphere_samples = util.sample_points_poisson_disk(sphere, self.sample_size)

        # find intersection with rays from the sphere of influence
        rays_to_center = sphere_center - sphere_samples
        (__, index_ray, ray_collission_on_object) = self.tri_mesh_obj.ray.intersects_id(
            ray_origins=sphere_samples, ray_directions=rays_to_center,
            return_locations=True, multiple_hits=False)
        self.np_src_cloud = np.empty((self.sample_size, 3))
        self.np_src_cloud[index_ray] = ray_collission_on_object

        # no intersection, then uses the nearest point in object
        no_index_ray = [i for i in range(self.sample_size) if i not in index_ray]
        (closest_in_obj, __, __) = self.tri_mesh_obj.nearest.on_surface(sphere_samples[no_index_ray])
        self.np_src_cloud[no_index_ray] = closest_in_obj

        return


class PropagateObjectNormalFromSpherePoissonDiscSamplerClearance(SamplerClearance):
    """
    Generates clearance vectors by
    1) sampling on a sphere of influence,
    2) generate rays from samples to the sphere centre
    3) find intersection of rays in object obtaining OBJECT SAMPLES
        IF NO INTERSECTION: find the nearest point from every "circle sample" in object
    4) calculate normal for every sample in object
    5) follow direction of normal until reaching IBS or the sphere of influence
    6) starting point of clearance vector in the sampling normal direction no further than threshold or IBS
    """
    influence_radio_ratio = 1.2
    distance_threshold = 0.05
    # work variables
    np_src_cloud_normal_vector = np.array([])

    def __init__(self, sample_size=None, distance_threshold=None):
        super().__init__()
        if sample_size is not None:
            self.sample_size = sample_size
        if distance_threshold is not None:
            self.distance_threshold = distance_threshold

    def get_source_cloud(self):
        sphere_ro, sphere_center = util.influence_sphere(self.tri_mesh_obj, self.influence_radio_ratio)
        sphere = trimesh.primitives.Sphere(radius=sphere_ro, center=sphere_center)
        sphere_samples = util.sample_points_poisson_disk(sphere, self.sample_size)

        # find intersection with rays from the sphere of influence
        rays_to_center = sphere_center - sphere_samples
        (index_triangle, index_ray, collision_point_on_object) = self.tri_mesh_obj.ray.intersects_id(
            ray_origins=sphere_samples, ray_directions=rays_to_center,
            return_locations=True, multiple_hits=False)
        # initialize work variables
        self.np_src_cloud = np.empty((self.sample_size, 3))
        self.np_src_cloud_normal_vector = np.empty((self.sample_size, 3))

        self.np_src_cloud[index_ray] = collision_point_on_object
        self.np_src_cloud_normal_vector[index_ray] = self.tri_mesh_obj.face_normals[index_triangle]

        # no intersection, then uses the nearest point in object
        no_index_ray = [i for i in range(self.sample_size) if i not in index_ray]
        (closest_in_obj, __, index_triangle) = self.tri_mesh_obj.nearest.on_surface(sphere_samples[no_index_ray])
        self.np_src_cloud[no_index_ray] = closest_in_obj
        self.np_src_cloud_normal_vector[no_index_ray] = self.tri_mesh_obj.face_normals[index_triangle]

        return

    def calculate_clearance_vectors(self):
        sphere_ro, sphere_center = util.influence_sphere(self.tri_mesh_obj, self.influence_radio_ratio)
        sphere = trimesh.primitives.Sphere(radius=sphere_ro, center=sphere_center)

        wrapper = self.tri_mesh_ibs + sphere

        (index_triangle, index_ray, locations) = wrapper.ray.intersects_id(
            ray_origins=self.np_src_cloud, ray_directions=self.np_src_cloud_normal_vector,
            return_locations=True, multiple_hits=False)

        raw_cv = self.np_src_cloud - locations
        norm_raw_cv = np.linalg.norm(raw_cv, axis=1)

        self.cv_points = np.zeros(locations.shape)
        self.cv_norms = np.zeros(norm_raw_cv.shape)

        idx_less_equal_than = norm_raw_cv <= self.distance_threshold
        idx_more_than = ~idx_less_equal_than

        # first assign all points in a SMALLER distance than the threshold
        self.cv_points[idx_less_equal_than] = locations[idx_less_equal_than]
        self.cv_norms[idx_less_equal_than] = norm_raw_cv[idx_less_equal_than]
        # adjust points to BIGGER distance than threshold
        self.cv_points[idx_more_than] = self.np_src_cloud[idx_more_than] + (
                    self.np_src_cloud_normal_vector[idx_more_than] * self.distance_threshold)
        self.cv_norms[idx_more_than] = self.distance_threshold

        self.cv_vectors = self.np_src_cloud - self.cv_points

    def get_info(self):
        info = super().get_info()
        info['influence_radio_ratio'] = self.influence_radio_ratio
        info['distance_threshold'] = self.distance_threshold
        return info

class PropagateNormalObjectPoissonDiscSamplerClearance(PropagateObjectNormalFromSpherePoissonDiscSamplerClearance):
    """
    Generates clearance vectors by
    1) poisson sampling on object,
    2) calculate normal for every sample in object
    3) follow direction of normal until reaching IBS or the sphere of influence
    4) starting point of clearance vector in the sampling normal direction no further than threshold or IBS
    5) vector goes from IBS or a point farther than distance_threshold to the sampling point in object
    """


    def get_source_cloud(self):
        self.np_src_cloud = util.sample_points_poisson_disk(self.tri_mesh_obj, self.sample_size)

        (closest_in_obj, __, index_triangle) = self.tri_mesh_obj.nearest.on_surface(self.np_src_cloud)

        self.np_src_cloud_normal_vector = self.tri_mesh_obj.face_normals[index_triangle]

        return
