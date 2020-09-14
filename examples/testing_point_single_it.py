import math
import numpy as np
import os

import trimesh
from scipy import linalg

from transforms3d.derivations.eulerangles import z_rotation
from transforms3d.affines import compose

from it_clearance.testing.tester import TesterClearance
from it_clearance.utils import central_point_mesh


def visual_comparison_of_trained_ibs_vs_ibs_on_point():
    from it.training.ibs import IBSMesh
    import it.util as util

    init_size_sampling = 400
    resamplings = 4

    print("calculating IBS surface un point and angle")
    ibs_calculator = IBSMesh(init_size_sampling, resamplings)
    obj_min_bound = np.asarray(tri_mesh_obj.vertices).min(axis=0)
    obj_max_bound = np.asarray(tri_mesh_obj.vertices).max(axis=0)
    extension = np.linalg.norm(obj_max_bound - obj_min_bound)
    middle_point = (obj_max_bound + obj_min_bound) / 2
    tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(tri_mesh_env, middle_point, extension)
    ibs_calculator.execute(tri_mesh_env_segmented, tri_mesh_obj)
    tri_mesh_ibs = ibs_calculator.get_trimesh()
    sphere_ro = np.linalg.norm(obj_max_bound - obj_min_bound)
    sphere_center = np.asarray(obj_max_bound + obj_min_bound) / 2
    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, sphere_center, sphere_ro)

    print("Done.\n visual comparison")
    tri_mesh_ibs_segmented.visual.face_colors = [0, 0, 255, 100]
    # tri_mesh_ibs_segmented.export("fedora_on_diffent_hanging_rack_ibs_mesh_segmented.ply", "ply")
    tri_mesh_ibs_segmented.show(caption='1/3 IBS surface in point', flags={'cull': False})

    # calculated IBS during training stage
    sub_working_path = working_directory + "/" + tester.affordances[0][0] + "/"
    ibs_file = sub_working_path + tester.affordances[0][0] + "_" + tester.affordances[0][1] + "_ibs_mesh_segmented.ply"
    trained_ibs = trimesh.load_mesh(ibs_file)
    trained_ibs.apply_transform(A)
    trained_ibs.visual.face_colors = [255, 0, 0, 100]
    trained_ibs.show(caption='2/3 Trained IBS surface', flags={'cull': False})

    # visualization of both IBS surfaces
    scene = trimesh.Scene([trained_ibs, tri_mesh_ibs_segmented])
    scene.show(caption='3/3 visual comparison of IBS surface', flags={'cull': False})


if __name__ == '__main__':
    working_directory = "./data/it/descriptors"

    tester = TesterClearance(working_directory, "./data/it/single_testing.json")

    tri_mesh_env = trimesh.load_mesh('./data/it/scene0000_00_vh_clean.ply', process=False)

    # testing_point = [3.73176694, 0.71387345, 0.53596354]  # touching wall collision with bed, angle 0
    testing_point = [3.19380641, 4.94784451, 0.52105212]  # touching couch collision with many objects
    # testing_point = [3.98721457, 2.82481766, 0.55264932]  # touching bed no collision, angle 0

    cv_analyzer = tester.get_analyzer_clearance(tri_mesh_env, testing_point)
    pv_analyzer = tester.get_analyzer(tri_mesh_env, testing_point)
    pv_analyzer.measure_scores()

    inter = 0  # it only works with one interaction
    print("interaction: ", tester.affordances[inter])
    print("{:<12} {:<14} {:<16} {:<16} {:<16}".format("orientation", "cv collisions", "%cv collisions",
                                                      "missing", "score"))
    for ori in range(tester.num_orientations):
        print("{:<12} {:<14} {:<16} {:<16} {:<16}".format(ori,
                                                  cv_analyzer.results.resumed_smaller_norm_by_inter_and_ori[inter][ori],
                                                  cv_analyzer.results.percentage_smaller_norm_by_inter_ori[inter][ori],
                                                  pv_analyzer.results.missed[inter][ori],
                                                  pv_analyzer.results.distances_summary[inter][ori]))

    ordered_idxs_lowest_possible_collision = np.argsort(cv_analyzer.results.resumed_smaller_norm_by_inter_and_ori[inter])

    for ori in range(tester.num_orientations):
        # analyses the first interaction (only one tested anyway)
        # take orientation with lowest collision possibilities
        orientation = ori
        angle = (2 * math.pi / tester.num_orientations) * orientation
        score = pv_analyzer.results.distances_summary[inter][ori]
        missing = pv_analyzer.results.missed[inter][ori]
        cv_collisions = cv_analyzer.results.resumed_smaller_norm_by_inter_and_ori[inter][ori]

        affordance_name = tester.affordances[0][0]
        affordance_object = tester.affordances[0][1]
        tri_mesh_object_file = tester.objs_filenames[0]
        influence_radius = tester.objs_influence_radios[0]

        # visualizing
        tri_mesh_obj = trimesh.load_mesh(tri_mesh_object_file, process=False)

        idx_from = orientation * tester.num_pv
        idx_to = idx_from + tester.num_pv
        pv_begin = tester.compiled_pv_begin[idx_from:idx_to]
        pv_direction = tester.compiled_pv_direction[idx_from:idx_to]
        provenance_vectors = trimesh.load_path(np.hstack((pv_begin, pv_begin + pv_direction)).reshape(-1, 2, 3))

        pv_intersections = pv_analyzer.calculated_pvs_intersection(0, orientation)

        R = z_rotation(angle)  # rotation matrix
        Z = np.ones(3)  # zooms
        T = testing_point
        A = compose(T, R, Z)
        tri_mesh_obj.apply_transform(A)

        tri_mesh_env.visual.face_colors = [100, 100, 100, 100]
        tri_mesh_obj.visual.face_colors = [0, 255, 0, 100]
        intersections = trimesh.points.PointCloud(pv_intersections, color=[0, 255, 255, 255])

        obj_centre = central_point_mesh(tri_mesh_obj)
        sphere = trimesh.primitives.Sphere(radius=influence_radius, center=obj_centre, subdivisions=4)
        sphere.visual.face_colors = [100, 0, 0, 20]

        scene = trimesh.Scene([provenance_vectors, intersections, tri_mesh_env, tri_mesh_obj, sphere])
        caption = 'Ori ' + str(ori) + ", cv_collisions: " + str(cv_collisions)
        caption += " ,score: " + str(score) + ", missing " + str(missing)
        scene.show(caption=caption)
        # visual_comparison_of_trained_ibs_vs_ibs_on_point()
        # tri_mesh_obj.apply_transform(linalg.inv(A))
