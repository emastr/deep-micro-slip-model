
import dolfin as dl
import numpy as np
from util.basis_scaled import ScaleShiftedBasis
import matplotlib.pyplot as plt

class Box(dl.SubDomain):
    def __init__(self, center: dl.Point, normal: dl.Point, width: float, height: float):
        """Make a box with center, normal, width and height"""
        self.center = center
        self.width = width
        self.height = height
        self.tangent = np.array([-normal[1], normal[0]])
        self.normal = normal
        super(Box, self).__init__()
        
    def inside(self, x, on_boundary):
        
        #return bool(max(abs(x[0]), abs(x[1])) < 0.4 + dl.DOLFIN_EPS and on_boundary)
        delta = x - self.center
        return bool( abs(np.dot(delta, self.tangent)) < self.width/2 - dl.DOLFIN_EPS  and
                    abs(np.dot(delta, self.normal)) < self.height/2 - dl.DOLFIN_EPS)
    
    def plot(self, ax):
        x1 = self.center + self.width/2 * self.tangent + self.height/2*self.normal
        x2 = self.center + self.width/2 * self.tangent - self.height/2*self.normal
        x3 = self.center - self.width/2 * self.tangent - self.height/2*self.normal
        x4 = self.center - self.width/2 * self.tangent + self.height/2*self.normal
        xs = [x1, x2, x3, x4, x1]
        ax.plot([x[0] for x in xs], [x[1] for x in xs])


def order_connected_vertices(mesh_1d):
    """Assuming vertices are connected, order them in the direction of the mesh"""
    mesh_1d.init(0, 1)
    visited = dl.MeshFunction("bool", mesh_1d, False)
    
    start_edge = dl.Edge(mesh_1d, 0)
    visited.array()[start_edge.index()] = True
    
    right_vert = dl.Vertex(mesh_1d, start_edge.entities(0)[1])
    left_vert = dl.Vertex(mesh_1d, start_edge.entities(0)[0])
    indices = [left_vert.index(), right_vert.index()]
    
    # Right search
    while len(right_vert.entities(1)) > 1:
        edge = dl.Edge(mesh_1d, [e for e in right_vert.entities(1) if not visited.array()[e]][0])
        right_vert = dl.Vertex(mesh_1d, [v for v in edge.entities(0) if v != right_vert.index()][0])
        indices = indices + [right_vert.index()]
        visited.array()[edge.index()] = True
    
    # Left search
    while len(left_vert.entities(1))> 1:
        edge = dl.Edge(mesh_1d, [e for e in left_vert.entities(1) if not visited.array()[e]][0])
        left_vert = dl.Vertex(mesh_1d, [v for v in edge.entities(0) if v != left_vert.index()][0])
        indices = [left_vert.index()] + indices
        visited.array()[edge.index()] = True
        
    return indices


def get_indicator(domain, space):
    """Get the indicator function of a domain"""
    indicator = dl.Function(space)
    indicator.vector()[:] = np.array([1. if domain.inside(x, False) else 0. for x in space.tabulate_dof_coordinates()])
    return indicator


def to_fenics_func(func, space):
    """Convert a function to a fenics function"""
    fenics_func = dl.Function(space)
    fenics_func.vector()[:] = np.array([func(x[0], x[1]) for x in space.tabulate_dof_coordinates()])
    return fenics_func


def boundary_bbox_tree(mesh: dl.Mesh, boundary: dl.SubDomain):
    bmesh = dl.BoundaryMesh(mesh, 'exterior')
    sbmesh = dl.SubMesh(bmesh, boundary)
    
    if sbmesh.num_cells() == 0:
        raise ValueError("No boundary found")
    
    else:
        bbtree = dl.BoundingBoxTree()
        bbtree.build(sbmesh)
    
        boundary_mesh_map = bmesh.entity_map(1)
        submesh_mesh_map = sbmesh.data().array('parent_cell_indices', 1)
        mesh_map = np.array([boundary_mesh_map[submesh_mesh_map[i]] for i in range(len(submesh_mesh_map))])
        return sbmesh, bbtree, mesh_map
    

def closest_boundary_facet(point, boundary_mesh, boundary_tree, boundary_mesh_map, mesh):
    bid, _ = boundary_tree.compute_closest_entity(point)
    facet = dl.Facet(mesh, boundary_mesh_map[bid])
    return facet


def closest_point_on_facet(point, facet):
    pts = [v.point() for v in dl.vertices(facet)]
    a, b = pts
    ab = b - a
    ax = point - a
    bx = point - b
    if ax.dot(ab) <= 0:
        return a
    elif -bx.dot(ab) <= 0:
        return b
    else:
        return a + ab * (ax.dot(ab) / ab.dot(ab))

    
def project_to_boundary(point, boundary_mesh, boundary_tree, boundary_mesh_map, mesh):
    facet = closest_boundary_facet(point, boundary_mesh, boundary_tree, boundary_mesh_map, mesh)
    point = closest_point_on_facet(point, facet).array()[:2]
    normal = facet.normal().array()[:2]
    return point, normal


def find_intersection(points, line_point, line_tangent):
    """Find intersection and return the index of point to the right of the intersection, 
    right in the direction of the tangent"""
    line_normal = np.array([line_tangent[1], -line_tangent[0]]) 
    
    for i in range(points.shape[0]-1):
        x0, x1 = points[i], points[i+1]
        dot_0 = np.dot(line_normal, x0 - line_point)
        dot_1 = np.dot(line_normal, x1 - line_point)
        if  dot_0 * dot_1 < 0:
            # find exact intersection
            t = abs(dot_0) / (abs(dot_1) + abs(dot_0))
            intersect = x0 * (1-t) + x1 * t
            # Return right point and intersection
            return intersect, i
        else:
            pass
    raise ValueError("No intersection found, pick a better box")


def find_intersection_on_segment(points, a, b, TOL=1e-8):
    """Find intersection and return the intersect point as well as the parameter t,
    which represents the relative distance from a to the intersection point,
    and return None if no intersection is found."""
    length = np.linalg.norm(b-a)
    normal = (b-a)/length
    intersect, _ = find_intersection(points, a, normal)

    t = np.dot(intersect - a, normal) / length

    print("Find intersection on segment doesn't check if the intersection is on the segment, it just clips it.", end='\r')
    if True:#t < 1 + TOL:# and t > 0 - TOL:
        t = max(min(t, 1.), 0.)
        return a*t + b*(1-t), t
    #else:
       #return None, None


def curve_length(curve):
    lengths = np.zeros(curve.shape[0])
    lengths[1:] = np.linalg.norm(curve[1:, :] - curve[:-1, :], axis=1)
    return np.cumsum(lengths)


def reparameterize_curve(curve, length, left_point, right_point, normal, dom):
    p_left, i_left = find_intersection(curve, left_point, normal)
    p_right, i_right = find_intersection(curve, right_point, normal)
    
    l_left = length[i_left] + np.linalg.norm(p_left - curve[i_left])
    l_right = length[i_right] + np.linalg.norm(p_right - curve[i_right])
    
    scale, shift = ScaleShiftedBasis._domain_to_scale_shift([l_left, l_right], dom)
    return scale*(length - shift)


def plot_boundary_mesh(mesh, **kwargs):
    for f in dl.cells(mesh):
        pts = [v.point() for v in dl.vertices(f)]
        plt.plot([pts[0].x(), pts[1].x()], [pts[0].y(), pts[1].y()], **kwargs)
    