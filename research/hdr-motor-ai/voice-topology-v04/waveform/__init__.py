"""HDR WaveForm v0.4 — research acoustics / 3D surface mechanics."""
from .core import Shape, geometry, signature, topology_2d
from .mesh import Mesh, make_mesh, deformed_vertices, export_obj
from .coupled import Wall, simulate, fit_shape
__all__ = ['Shape','geometry','signature','topology_2d','Mesh','make_mesh','deformed_vertices','export_obj','Wall','simulate','fit_shape']
