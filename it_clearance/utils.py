
from vtkplotter import Lines, Spheres, trimesh2vtk, Plotter
from vtkplotter.utils import flatten


def get_vtk_plotter_cv_pv(pv_points, pv_vectors, cv_points, cv_vectors,
                         trimesh_env=None, trimesh_obj=None, trimesh_ibs=None):

    clearance_vectors = Lines(cv_points, cv_points + cv_vectors, c='yellow', alpha=1).lighting("plastic")
    provenance_vectors = Lines(pv_points, pv_points + pv_vectors, c='red', alpha=1).lighting("plastic")
    cv_from = Spheres(cv_points, r=.007, c="yellow", alpha=1).lighting("plastic")

    plot_elements = [clearance_vectors, provenance_vectors, cv_from]

    if trimesh_env is not None:
        trimesh_env.visual.face_colors = [200, 200, 200, 255]
        vtk_env = trimesh2vtk(trimesh_env)
        vtk_env.lighting("plastic")
        plot_elements.append(vtk_env)

    if trimesh_obj is not None:
        trimesh_obj.visual.face_colors = [0, 250, 0, 254]
        vtk_obj = trimesh2vtk(trimesh_obj)
        vtk_obj.lighting("plastic")
        plot_elements.append(vtk_obj)

    if trimesh_ibs is not None:
        trimesh_ibs.visual.face_colors = [0, 0, 200, 100]
        vtk_ibs = trimesh2vtk(trimesh_ibs)
        vtk_ibs.lighting("plastic")
        plot_elements.append(vtk_ibs)

    plot_elements = flatten(plot_elements)

    vp = Plotter(bg="white")
    vp.add(plot_elements)
    return vp
