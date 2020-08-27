import math
import numpy as np
import os

import trimesh
from scipy import linalg

from transforms3d.derivations.eulerangles import z_rotation
from transforms3d.affines import compose

from it_clearance.testing.tester import TesterClearance


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

    tri_mesh_env = trimesh.load_mesh('./data/it/gates400.ply', process=False)
    testing_point = [-0.48689266781021423, -0.15363679409350514, 0.8177121144402457]
    # testing_point = [-0.97178262, -0.96805501, 0.82738298] #in the edge of table, but with floor
    # testing_point = [-2.8, 1., 0.00362764]  # half inside the scene, half outside

    cv_analyzer = tester.get_analyzer_clearance(tri_mesh_env, testing_point)

    for inter in range(tester.num_it_to_test):
        print("interaction: ", tester.affordances[inter])
        for ori in range(tester.num_orientations):
            print("   orientation: ", ori,
                  " possible collisions: ", cv_analyzer.results.resumed_smaller_norm_by_inter_and_ori[inter][ori],
                  " percentage pos collisions: ", cv_analyzer.results.percentage_smaller_norm_by_inter_ori[inter][ori])

    pv_analyzer = tester.get_analyzer(tri_mesh_env, testing_point)

    angles_with_best_score = pv_analyzer.best_angle_by_distance_by_affordance()
    all_distances, resumed_distances, missed = pv_analyzer.measure_scores()

    # as this is a run with only one affordance to test, only get the first row of results
    first_affordance_scores = angles_with_best_score[0]
    orientation = int(first_affordance_scores[0])
    angle = first_affordance_scores[1]
    score = first_affordance_scores[2]
    missing = first_affordance_scores[3]

    print("score: " + str(score) + ", missing " + str(missing))

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

    sphere = trimesh.primitives.Sphere(radius=influence_radius, center=testing_point, subdivisions=4)
    sphere.visual.face_colors = [100, 0, 0, 20]

    scene = trimesh.Scene([provenance_vectors, intersections, tri_mesh_env, tri_mesh_obj, sphere])
    scene.show()
    # visual_comparison_of_trained_ibs_vs_ibs_on_point()
    # tri_mesh_obj.apply_transform(linalg.inv(A))
