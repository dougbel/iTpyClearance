import math

import numpy as np
import vtk
from prettytable import PrettyTable
from vedo import load, Text2D, Plotter, Point, Lines, Spheres, Points
from vedo.colors import colorMap
from vedo.utils import vtk2trimesh

from it_clearance.testing.tester import TesterClearance


def on_left_click(mesh):
    global np_testing_point, vedo_testing_point
    if activated:
        if vedo_testing_point is not None:
            vp.clear(vedo_testing_point)
        np_testing_point = mesh.picked3d
        vedo_testing_point = Point(np_testing_point, c='red')
        vp.add(vedo_testing_point)
        test_on_clicked_point()


def on_key_press(key):
    global activated, vp
    if key == 'c':
        activated = not activated
        if activated:
            vp.clear()
            vp.add(vedo_env)
            vp.add(p)
            vp.add(txt_enable)
        else:
            vp.clear(txt_enable)
            vp.add(txt_disable)


def test_on_clicked_point():
    global tester, vedo_env, first_test

    txt = Text2D("=== PROCESSING ===", pos="bottom-left", s=1, c='darkred', bg="white",
                 font='ImpactLabel', justify='center')
    vp.add(txt)

    cv_analyzer = tester.get_analyzer_clearance(tri_mesh_env_filled, np_testing_point)
    pv_analyzer = tester.get_analyzer(tri_mesh_env, np_testing_point)
    pv_analyzer.measure_scores()
    inter = 0  # #################################################################### it ONLY works with one interaction
    print("interaction: ", tester.affordances[inter])

    table = PrettyTable()

    table.field_names = ["orientation", "cv collisions", "%cv collisions", "missing", "score"]

    for ori in range(tester.num_orientations):
        table.add_row([ori,
                       cv_analyzer.results.resumed_smaller_norm_by_inter_and_ori[inter][ori],
                       cv_analyzer.results.percentage_smaller_norm_by_inter_ori[inter][ori],
                       pv_analyzer.results.missed[inter][ori],
                       pv_analyzer.results.distances_summary[inter][ori]]
                      )
    print(table)

    ordered_idxs_lowest_possible_collision = np.argsort(
        cv_analyzer.results.resumed_smaller_norm_by_inter_and_ori[inter])

    angle = (2 * math.pi / tester.num_orientations)
    for ori in range(tester.num_orientations):
        idx_from = ori * tester.num_pv
        idx_to = idx_from + tester.num_pv
        pv_points = tester.compiled_pv_begin[idx_from:idx_to]
        pv_vectors = tester.compiled_pv_direction[idx_from:idx_to]
        score = pv_analyzer.results.distances_summary[inter][ori]
        cv_collisions = cv_analyzer.results.resumed_smaller_norm_by_inter_and_ori[inter][ori]

        idx_from = ori * tester.num_cv
        idx_to = idx_from + tester.num_cv
        cv_points = tester.compiled_cv_begin[idx_from:idx_to]
        cv_vectors = tester.compiled_cv_direction[idx_from:idx_to]

        provenance_vectors = Lines(pv_points, pv_points + pv_vectors, c='red', alpha=1).lighting("plastic")
        clearance_vectors = Lines(cv_points, cv_points + cv_vectors, c='yellow', alpha=1).lighting("plastic")
        cv_from = Spheres(cv_points, r=.007, c="yellow", alpha=1).lighting("plastic")

        vedo_obj = load(tester.objs_filenames[0]).lighting("plastic")
        vedo_obj.c(colorMap(score, name='Blues_r', vmin=min_score, vmax=max_score))
        max_collisions = 2
        if cv_collisions > max_collisions:
            vedo_obj.alpha(.05)
            provenance_vectors.alpha(.05)
            clearance_vectors.alpha(.05)
            cv_from.alpha(.05)
        else:
            a = max([1-cv_collisions*(1/(max_collisions+1)), .05])
            vedo_obj.alpha(a)
            provenance_vectors.alpha(a)
            clearance_vectors.alpha(a)
            cv_from.alpha(a)

        vedo_obj.rotateZ(angle*ori, rad=True)
        vedo_obj.pos(x=np_testing_point[0], y=np_testing_point[1], z=np_testing_point[2])

        vp.add(provenance_vectors)
        vp.add(clearance_vectors)
        vp.add(cv_from)
        vp.add(vedo_obj)

    vp.clear(txt)
    on_key_press("c")



if __name__ == '__main__':
    activated = False
    np_testing_point = None
    vedo_testing_point = None
    first_test = True

    working_directory = "./data/it/descriptors"
    config_file = "./data/it/single_testing.json"
    env_file = './data/it/scene0000_00_vh_clean.ply'
    env_file_filled = './data/it/filled_scene0000_00_vh_clean.ply'

    max_score = 75.41
    min_score = 0
    max_limit_missing = 36


    tester = TesterClearance(working_directory, config_file)

    vedo_env = load(env_file).lighting("plastic")
    vedo_env_filled = load(env_file_filled).lighting("plastic")

    tri_mesh_env = vtk2trimesh(vedo_env)
    tri_mesh_env_filled = vtk2trimesh(vedo_env_filled)

    txt_enable = Text2D('Left click selection enabled ("c")', pos='bottom-right', c='steelblue', bg='black',
                        font='ImpactLabel', alpha=1)
    txt_disable = Text2D('Left click selection disabled ("c")', pos='bottom-right', c='darkred', bg='black',
                         font='ImpactLabel', alpha=1)

    vp = Plotter(verbose=0, pos=(250, 0), size=(860,860))
    vp.mouseLeftClickFunction = on_left_click
    vp.keyPressFunction = on_key_press
    p = Points(np.zeros((tester.num_orientations, 3)), r=1, alpha=0, c='blue')
    values = [((max_score - min_score) / (tester.num_orientations-1))*i for i in range(tester.num_orientations)]
    p.cellColors(values, cmap='Blues_r', vmin=min_score, vmax=max_score)
    p.addScalarBar(nlabels=5)

    vp.show(vedo_env, txt_disable, p)
