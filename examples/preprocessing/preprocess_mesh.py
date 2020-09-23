import logging

import numpy as np
from vedo import Plotter, write, merge

from it_clearance.preprocessing.bubble_filler import BubbleFiller


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    env_file = "./data/it/scene0000_00_vh_clean.ply"

    filler = BubbleFiller(env_file)
    fine_bubbles = filler.calculate_fine_bubble_filler(0.03)
    gross_bubbles = filler.calculate_gross_bubble_filler(0.07)
    floor_filler = filler.calculate_floor_holes_filler(0.12)

    # collision_tester = trimesh.collision.CollisionManager()
    # collision_tester.add_object("environment", filler.tri_mesh_env)
    vp = Plotter(bg="white")
    vp.add(filler.vedo_env)
    # vp.add(Points(seed_small_points,  c=fine_bubbles.c()))
    vp.add(fine_bubbles)
    # vp.add(Points(seed_gross_points, c=gross_bubbles.c()))
    vp.add(gross_bubbles)

    vp.add(floor_filler)

    write(fine_bubbles, "./output/filler_fine_bubbles.ply")
    write(gross_bubbles, "./output/filler_gross_bubbles.ply")
    write(floor_filler, "./output/filler_floor_holes.ply")
    write(merge(filler.vedo_env, gross_bubbles, floor_filler, ), "./output/filled_env.ply")

    vp.show()
