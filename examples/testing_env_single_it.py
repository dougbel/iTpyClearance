import os
import time

import trimesh

import it.util as util
from it_clearance.testing.envirotester import EnviroTesterClearance

if __name__ == '__main__':

    tri_mesh_env = trimesh.load_mesh('./data/it/scene0000_00_vh_clean.ply')

    start = time.time()  # timing execution
    np_test_points, np_env_normals = util.sample_points_poisson_disk_radius(tri_mesh_env, radius=0.1)
    end = time.time()  # timing execution
    print("Sampling 1 Execution time: ", end-start)

    start = time.time()  # timing execution
    sampling_size = np_test_points.shape[0]
    np_test_points = util.sample_points_poisson_disk(tri_mesh_env, sampling_size)
    np_env_normals = util.get_normal_nearest_point_in_mesh(tri_mesh_env, np_test_points)
    end = time.time()  # timing execution
    print("Sampling 2 Execution time: ", end - start)


    # Load configurations for ONE interaction test
    directory_of_trainings = "./data/it/descriptors"
    json_conf_execution_file = "./data/it/single_testing.json"

    tester = EnviroTesterClearance(directory_of_trainings, json_conf_execution_file)

    affordance_name = tester.affordances[0][0]
    affordance_object = tester.affordances[0][1]
    tri_mesh_object_file = tester.objs_filenames[0]

    tri_mesh_obj = trimesh.load_mesh(tri_mesh_object_file)

    start = time.time()  # timing execution
    # Testing iT
    full_data_frame = tester.start_full_test(tri_mesh_env, np_test_points, np_env_normals)
    end = time.time()  # timing execution
    print("Testing execution time: ", end - start)

    end = time.time()  # timing execution
    print("Execution time: ", end-start)


    # ##################################################################################################################
    # SAVING output
    output_dir = './output/testing_env_single'
    output_dir = os.path.join(output_dir, affordance_name + '_' + affordance_object, )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    full_data_frame.to_csv(os.path.join(output_dir, "test_scores.csv"))