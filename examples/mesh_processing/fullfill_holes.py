
import trimesh
import gc

import numpy as np

from sklearn.preprocessing import normalize
from vedo import load, Plotter, Spheres, Lines

if __name__ == '__main__':

    a = load("./data/it/small_scene0000_00_vh_clean.ply")

    # size = approximate limit to the size of the hole to be filled.
    b = a.clone().fillHoles(size=1)
    b.color("red").legend("filled mesh").c("gray").bc("t")
    vp = Plotter(bg="white")
    vp.show(b,a)