import numpy as np
from matplotlib.path import Path

def triangle_area(vertices):
    """
    Calculate the area of a triangle given its 3 vertices (2D or 3D).
    """
    v = np.asarray(vertices)
    if v.shape[1] == 2:
        x1, y1 = v[0]
        x2, y2 = v[1]
        x3, y3 = v[2]
        area = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
    elif v.shape[1] == 3:
        a = v[1] - v[0]
        b = v[2] - v[0]
        area = 0.5 * np.linalg.norm(np.cross(a, b))
    else:
        raise ValueError("vertices must be shape (3, 2) or (3, 3)")
    return area

def triangle_to_rectangles(vertices, dx=0.1, dy=0.1, area_threshold=1/2):
    """
    Cover a triangle with equally spaced rectangular elements and filter out valid elements.
    """
    x_min, y_min = np.min(vertices, axis=0)
    x_max, y_max = np.max(vertices, axis=0)
    xs = np.arange(x_min, x_max+dx/2.0, dx)
    ys = np.arange(y_min, y_max+dy/2.0, dy)
    rects = []
    centers = []
    buffer_dist = (dx + dy) / 4.0

    from shapely.geometry import Polygon, Point
    tri_poly = Polygon(vertices)
    tri_buffer = tri_poly.buffer(buffer_dist)
    for x0 in xs:
        for y0 in ys:
            rect = np.array([
                [x0, y0], # Bottom-Left
                [x0+dx, y0], # Bottom-Right
                [x0+dx, y0+dy], # Top-Right
                [x0, y0+dy] # Top-Left
            ])
            center = np.mean(rect, axis=0)
            pt = Point(center)
            if tri_poly.contains(pt):
                rects.append(rect)
                centers.append(center)
            elif tri_buffer.contains(pt):
                rect_poly = Polygon(rect)
                inter_area = tri_poly.intersection(rect_poly).area
                if inter_area >= area_threshold * dx * dy:
                    rects.append(rect)
                    centers.append(center)
    rect_corners = rects
    rect_centers = np.array(centers)
    return rect_centers, rect_corners

def patch_local2d(vertices_3d, cx_km, cy_km, depth_km, strike_rad, dip_rad):
    """
    Convert 3D triangle vertices to local 2D coordinates on the fault plane.
    """
    v = vertices_3d - np.array([cx_km, cy_km, depth_km])[np.newaxis, :]
    strike_rad = np.pi/2.0 - strike_rad
    vert_xy = (v[:, 0] + 1.j*v[:, 1]) * np.exp(-1.j*strike_rad)
    x, y = vert_xy.real, vert_xy.imag
    vert_yz = (y + 1.j*v[:, 2]) * np.exp(1.j*dip_rad)
    xyz = np.column_stack([x, vert_yz.real, vert_yz.imag])
    return xyz

def patch_local2d_inv(xyz, cx_km, cy_km, depth_km, strike_rad, dip_rad):
    """
    Convert local 2D (or 3D) coordinates on the fault plane back to original 3D coordinates.
    """
    strike_rad = np.pi/2.0 - strike_rad
    rc_yz = (xyz[:, 1] + 1.j*xyz[:, 2]) * np.exp(-1.j*dip_rad)
    rc_y, rc_z = rc_yz.real, rc_yz.imag
    rc_xy = (xyz[:, 0] + 1.j*rc_y) * np.exp(1.j*strike_rad)
    rc_x, rc_y = rc_xy.real, rc_xy.imag
    return np.column_stack([rc_x, rc_y, rc_z]) + np.array([cx_km, cy_km, depth_km])[np.newaxis, :]

def plot_triangle_and_rects_2d(vertices_transfer, rect_corners):
    """
    Plot triangle and rectangles in 2D.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,6))
    tri = np.vstack([vertices_transfer, vertices_transfer[0]])
    ax.plot(tri[:,0], tri[:,1], 'k-', lw=2, label='Triangle')
    for corners in rect_corners:
        rect2d = corners[:4, :2]
        rect2d = np.vstack([rect2d, rect2d[0]])
        ax.plot(rect2d[:,0], rect2d[:,1], 'b-', alpha=0.5)
        ax.plot(np.mean(rect2d[:-1,0]), np.mean(rect2d[:-1,1]), 'ro', ms=2)
    ax.set_aspect('equal')
    ax.set_title('Triangle and Equivalent Rectangular Elements (xy projection)')
    ax.legend()
    plt.show()

def plot_triangle_and_rects_3d(vertices_3d_orig, vertices_3d, rect_corners_3d):
    """
    Plot triangle and rectangles in 3D.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    tri3d = np.vstack([vertices_3d_orig, vertices_3d_orig[0]])
    ax3d.plot(tri3d[:,0], tri3d[:,1], tri3d[:,2], 'r-', lw=2, label='Triangle Orig')
    tri3d = np.vstack([vertices_3d, vertices_3d[0]])
    ax3d.plot(tri3d[:,0], tri3d[:,1], tri3d[:,2], 'k-', lw=2, label='Triangle')
    for icorners in rect_corners_3d:
        face = icorners
        poly3d = Poly3DCollection([face], facecolors='cyan', edgecolors='b', linewidths=0.5, alpha=0.4)
        ax3d.add_collection3d(poly3d)
        center = np.mean(face, axis=0)
        ax3d.scatter(center[0], center[1], center[2], color='r', s=5)
    ax3d.set_xlabel('X (km)')
    ax3d.set_ylabel('Y (km)')
    ax3d.set_zlabel('Z (km)')
    ax3d.set_title('Triangle and Equivalent Rectangular Elements (3D)')
    ax3d.legend()
    plt.show()

if __name__ == "__main__":
    from csi.TriangularPatches import TriangularPatches
    from csi.RectangularPatches import RectangularPatches
    lon0 = 120.5
    lat0 = 30.0
    myrect = RectangularPatches('Taiwan', lon0=lon0, lat0=lat0)
    myrect.readPatchesFromFile('slip_taiwan_Main.gmt', readpatchindex=True, )
    mytri = TriangularPatches('Taiwan', lon0=lon0, lat0=lat0)
    mytri.patches2triangles(fault=myrect, numberOfTriangles=2)

    # Get patch 0
    p = 0
    vertices_3d = mytri.patch[p]
    vertices_3d_orig = vertices_3d.copy()
    print(vertices_3d_orig)
    vertices = vertices_3d[:, :2]

    # Get geometry and convert to local 2D
    cx_km, cy_km, depth_km, width_km, length_km, strike_rad, dip_rad = mytri.getpatchgeometry(p, center=True)
    print(cx_km, cy_km, depth_km, width_km, length_km, strike_rad, dip_rad)
    xyz = patch_local2d(vertices_3d_orig, cx_km, cy_km, depth_km, strike_rad, dip_rad)
    vertices_transfer = xyz[:, :2]

    dx = dy = 0.1  # 100m
    rect_centers, rect_corners = triangle_to_rectangles(xyz[:, :2], dx, dy)

    # 2D plot
    plot_triangle_and_rects_2d(vertices_transfer, rect_corners)

    # Recover to original 3D coordinates
    rect_corners_3d = [
        patch_local2d_inv(
            np.column_stack([c, np.zeros((c.shape[0], 1))]),
            cx_km, cy_km, depth_km, strike_rad, dip_rad
        )
        for c in rect_corners
    ]
    rect_centers_3d = np.array([np.mean(c, axis=0) for c in rect_corners_3d])
    vertices_3d = patch_local2d_inv(xyz, cx_km, cy_km, depth_km, strike_rad, dip_rad)
    print(vertices_3d)
    # 3D plot
    plot_triangle_and_rects_3d(vertices_3d_orig, vertices_3d, rect_corners_3d)