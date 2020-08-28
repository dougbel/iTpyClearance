import trimesh
import numpy as np
import open3d as o3d
from transforms3d.affines import compose
import pandas as pd
import math
import time
import os
from it_clearance.testing.tester import Tester
import it_clearance.util as util

last_position = np.array([0, 0, 0])
index = 0
r = np.eye(3)  # rotation matrix
z = np.ones(3)  # zooms
MAX_SCORE_GOOD_POINT = 2 #REVIEW This parameter was setted only for place-bowl interaction


def move_object(dynamic_scene):
    """
    A callback passed to a scene viewer which will update
    transforms in the viewer periodically.
    Parameters
    -------------
    dynamic_scene : trimesh.Scene
      Scene containing geometry
    """
    global last_position
    global index

    translation = np_test_points[index] - last_position
    index += 1

    t = translation  # translations

    a = compose(t, r, z)

    # take one of the two spheres arbitrarily
    node = dynamic_scene.graph.nodes_geometry[1]
    # apply the transform to the node
    dynamic_scene.graph.update(node, matrix=a)


def test_collision(tri_mesh_env, tri_mesh_obj, points_to_test):
    no_collided = []
    last_pos = np.array([0, 0, 0])
    index = 0
    progress = 0
    period = []

    # TODO check if exists a faster lib
    collision_tester = trimesh.collision.CollisionManager()
    collision_tester.add_object('scene', tri_mesh_env)
    for point in points_to_test:
        translation = point - last_pos

        tri_mesh_obj.apply_translation(translation)

        '''tri_mesh_obj.visual.face_colors = [0, 255, 0, 255]
        tri_mesh_env.visual.face_colors = [100, 100, 100, 100]
        output = trimesh.points.PointCloud(points_to_test, colors=[255, 0, 0, 100])
        scene = trimesh.Scene([tri_mesh_env, tri_mesh_obj, output])
        scene.show()'''

        # TODO TEST WITH Open3D is_intersecting(self, arg0)
        #  http://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html
        start = time.time()  # timing execution
        in_collision = collision_tester.in_collision_single(tri_mesh_obj)
        end = time.time()  # timing execution
        period.append(end - start)  # timing execution

        if not in_collision:
            no_collided.append(point)
        last_pos = point

        current_percent = int(100 * index / points_to_test.shape[0])
        if current_percent - progress > 0:
            progress = current_percent
            print(progress, "%")
        index += 1

    print(np.asarray(period).mean(), " seconds on each collision test")  # timing execution
    print(60 / np.asarray(period).mean(), " collision test each min")  # timing execution
    return no_collided


def test_it(it_tester, environment, points_to_test, np_env_normals):
    idx = 0
    progress = 0
    period = []

    data_frame = pd.DataFrame(columns=['point_x', 'point_y', 'point_z','point_nx', 'point_ny', 'point_nz', 'diff_ns' , 'best_score', 'missings', 'best_angle',
                                       'best_orientation', 'calculation_time'])

    l_good_points = []

    #for testing_point in points_to_test:
    for i in range(points_to_test.shape[0]):
        testing_point = points_to_test[i]
        env_normal = np_env_normals[i]

        start = time.time()  # timing execution

        normals_angle = util.angle_between(it_tester.envs_normals[0], env_normal)

        if normals_angle > math.pi/3:
            orientation = math.nan
            angle = math.nan
            score = math.nan
            missing = math.nan
        else:
            analyzer = it_tester.get_analyzer(environment, testing_point)
            angle_with_best_score = analyzer.best_angle_by_distance_by_affordance()

            first_affordance_scores = angle_with_best_score[0]
            orientation = int(first_affordance_scores[0])
            angle = first_affordance_scores[1]
            score = first_affordance_scores[2]
            missing = int(first_affordance_scores[3])

        end = time.time()  # timing execution
        calculation_time = end - start
        period.append(calculation_time)  # timing execution

        data_frame.loc[len(data_frame)] = [testing_point[0], testing_point[1], testing_point[2],
                                           env_normal[0], env_normal[1], env_normal[2],
                                           normals_angle, score, missing, angle,
                                           orientation, calculation_time]

        if score < MAX_SCORE_GOOD_POINT:
            l_good_points.append(testing_point)

        current_percent = int(100 * idx / points_to_test.shape[0])
        if current_percent - progress > 0:
            progress = current_percent
            print(progress, "%")
        idx += 1

    print(np.asarray(period).mean(), " seconds on iT test")  # timing execution
    print(60 / np.asarray(period).mean(), " iT tests each min")  # timing execution

    np_good_points = np.asarray(l_good_points)

    return data_frame, np_good_points


if __name__ == '__main__':
    # TODO Measure the maximum distance by influence radius
    # TODO Test avoiding points with normal vector in environment
    # TODO Test with the kitchen (artificial scene)
    # Try different combinations of distances and missing values

    sampling_size = 163058
    tri_mesh_env = trimesh.load_mesh('./data/it/gates400.ply')

    # Load configurations for ONE interaction test
    directory_of_trainings = "./data/it/IBSMesh_400_4_OnGivenPointCloudWeightedSampler_5_500"
    json_conf_execution_file = directory_of_trainings + "/single_testing.json"

    tester = Tester(directory_of_trainings, json_conf_execution_file)

    affordance_name = tester.affordances[0][0]
    affordance_object = tester.affordances[0][1]
    tri_mesh_object_file = tester.objs_filenames[0]

    tri_mesh_obj = trimesh.load_mesh(tri_mesh_object_file)

    start = time.time()  # timing execution
    #np_test_points = util.sample_points_poisson_disk(tri_mesh_env, sampling_size)
    #np_env_normals = util.get_normal_nearest_point_in_mesh(tri_mesh_env, np_test_points)

    np_test_points, np_env_normals = util.sample_points_poisson_disk_radius(tri_mesh_env, radius=0.01)

    # Testing iT
    results_it_test, good_points = test_it(tester, tri_mesh_env, np_test_points, np_env_normals)

    # Execution time sample_points_poisson_disk_radius:  1405.92 with 163,058 points
    # Execution time sample_points_poisson_disk:         1348.92 with 163,058 points

    end = time.time()  # timing execution
    print("Execution time: ", end-start)

    # Testing collision
    no_collision = test_collision(tri_mesh_env, tri_mesh_obj, np_test_points)

    # ##################################################################################################################
    # SAVING output
    output_dir = './output/testing_env_single'
    output_dir = os.path.join(output_dir, affordance_name + '_' + affordance_object, )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tri_mesh_env.export(output_dir + "/environment.ply", "ply")
    tri_mesh_obj.export(output_dir + "/object.ply", "ply")
    tri_mesh_obj.visual.face_colors = [0, 255, 0, 255]
    tri_mesh_env.visual.face_colors = [100, 100, 100, 100]

    # test points
    o3d_test_points = o3d.geometry.PointCloud()
    o3d_test_points.points = o3d.utility.Vector3dVector(np_test_points)
    o3d.io.write_point_cloud(output_dir + "/test_points.pcd", o3d_test_points)

    # it test
    filename = "%s/scores.csv" % output_dir
    results_it_test.to_csv(filename)
    good_points_pcd = o3d.geometry.PointCloud()
    good_points_pcd.points = o3d.utility.Vector3dVector(good_points)
    o3d.io.write_point_cloud(output_dir + "/good_points.pcd", good_points_pcd)
    output = trimesh.points.PointCloud(np.asarray(good_points))
    scene = trimesh.Scene([tri_mesh_env, tri_mesh_obj, output])
    scene.show(caption="Score < "+str(MAX_SCORE_GOOD_POINT))

    # collision test
    no_collision_pcd = o3d.geometry.PointCloud()
    no_collision_pcd.points = o3d.utility.Vector3dVector(no_collision)
    o3d.io.write_point_cloud(output_dir + "/no_collision_points.pcd", no_collision_pcd)

    output = trimesh.points.PointCloud(np.asarray(no_collision))
    scene = trimesh.Scene([tri_mesh_env, tri_mesh_obj, output])
    scene.show(caption="No colission positions")

    # environment.show(callback=move_object)

    # TODO use libraries as cupy to work with numpy arrays
    # TODO fin alternative ray tracing approaches that uses the gpu
    # TODO generate and try scores
