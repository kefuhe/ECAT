from shapely.geometry import LineString, Point
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PolygonIntersector:
    def __init__(self, top_coords, bottom_coords, depth=18.0, extension_length=1000):
        """
        初始化多边形交点计算器。
        
        :param top_coords: 顶边坐标列表。
        :param bottom_coords: 底边坐标列表。
        :param depth: 深度值，用于计算交点的Z坐标。
        :param extension_length: 线段外延长度。
        """
        self.top_coords = top_coords.tolist() if isinstance(top_coords, np.ndarray) else top_coords
        self.bottom_coords = bottom_coords.tolist() if isinstance(bottom_coords, np.ndarray) else bottom_coords
        self.depth = depth
        self.extension_length = extension_length

    def calculate_normal_vector(self, coords, index):
        """
        计算给定坐标点的法向量。
        """
        if index == 0:
            dir_vector = np.array(coords[1]) - np.array(coords[0])
        elif index == len(coords) - 1:
            dir_vector = np.array(coords[-1]) - np.array(coords[-2])
        else:
            dir_vector = np.array(coords[index + 1]) - np.array(coords[index])
        normal_vector = np.array([-dir_vector[1], dir_vector[0], 0])
        return normal_vector

    def find_intersection(self, top_point, index):
        """
        查找给定顶点和其法线与底边折线的交点。
        """
        top_normal_vector = self.calculate_normal_vector(self.top_coords, index)
        P0 = np.array(top_point[:2])
        dir = top_normal_vector[:2] / np.linalg.norm(top_normal_vector[:2])
        line_top = LineString([P0 + dir * -self.extension_length, P0, P0 + dir * self.extension_length])
        
        extended_bottom_coords = self._extend_bottom_line()
        line_bottom = LineString(extended_bottom_coords)
        
        intersection = line_top.intersection(line_bottom)
        return self._process_intersection(intersection, P0)

    def _extend_bottom_line(self):
        """
        拓展底边折线两端。
        """
        bottom_start = np.array(self.bottom_coords[0])
        bottom_end = np.array(self.bottom_coords[-1])
        bottom_dir_start = np.array(self.bottom_coords[1]) - bottom_start
        bottom_dir_end = np.array(self.bottom_coords[-2]) - bottom_end
        bottom_dir_start /= np.linalg.norm(bottom_dir_start)
        bottom_dir_end /= np.linalg.norm(bottom_dir_end)
        
        extended_start = bottom_start + bottom_dir_start * -self.extension_length
        extended_end = bottom_end + bottom_dir_end * -self.extension_length
        return [extended_start] + self.bottom_coords + [extended_end]

    def _process_intersection(self, intersection, P0):
        """
        处理交点结果。
        """
        P0_point = Point(P0)
        if not intersection.is_empty:
            if intersection.geom_type == 'MultiPoint':
                closest_point = min([p for p in intersection.geoms], key=lambda p: P0_point.distance(p))
                return np.array([closest_point.x, closest_point.y, -self.depth])
            else:
                return np.array([intersection.x, intersection.y, -self.depth])
        return None

    def calculate_intersections(self, mode='both', indices=None):
        """
        计算所有顶点与底边折线的交点。
        参数:
        - mode: 'all', 'first', 'last', 或 'specific'，默认为 'all'。
        - indices: 当 mode 为 'specific' 时，指定需要计算交点的顶点索引列表。
        """
        intersections = []
        if mode == 'all':
            indices_to_check = range(len(self.top_coords))
        elif mode == 'first':
            indices_to_check = [0]
        elif mode == 'last':
            indices_to_check = [len(self.top_coords) - 1]
        elif mode == 'both':
            indices_to_check = [0, len(self.top_coords) - 1]
        elif mode == 'specific' and indices is not None:
            indices_to_check = indices
        else:
            raise ValueError("Invalid mode or indices")
    
        for index in indices_to_check:
            top_point = self.top_coords[index]
            intersection = self.find_intersection(top_point, index)
            if intersection is not None:
                intersections.append(intersection)
        return intersections

    def plot(self, style=['notebook'], plot_on_2d=False):
        """
        绘制顶边、底边和交点。
        """
        from ..plottools import sci_plot_style
        with sci_plot_style(style=style):
            fig = plt.figure()
            if plot_on_2d:
                ax = fig.add_subplot(111)
            else:
                ax = fig.add_subplot(111, projection='3d')
            
            top_x, top_y, top_z = zip(*self.top_coords)
            bottom_x, bottom_y, bottom_z = zip(*self.bottom_coords)
            intersections = self.calculate_intersections()

            # 绘制顶边和底边
            if plot_on_2d:
                ax.plot(top_x, top_y, color='r', marker='o', label='Top Edge')
                ax.plot(bottom_x, bottom_y, color='b', marker='o', label='Bottom Edge')
            else:
                ax.plot(top_x, top_y, top_z, color='r', marker='o', label='Top Edge')
                ax.plot(bottom_x, bottom_y, bottom_z, color='b', marker='o', label='Bottom Edge')
            
            # 收集交点坐标
            intersections_x = [point[0] for point in intersections]
            intersections_y = [point[1] for point in intersections]
            if not plot_on_2d:
                intersections_z = [point[2] for point in intersections]
            
            # 使用plot命令绘制交点
            if plot_on_2d:
                ax.plot(intersections_x, intersections_y, 'g^', label='Intersection', zorder=3)
            else:
                ax.plot(intersections_x, intersections_y, intersections_z, 'g^', label='Intersection', zorder=3)
            
            ax.legend()
            plt.show()


if __name__ == '__main__':
    # 示例使用
    top_coords = np.array([(0, 0, 0), (1, 1, 0), (2, 2, 0)])  # 三个或更多节点
    bottom_coords = np.array([(0, -1, -18), (1.5, 1, -18), (2.3, 1, -18), ])
    patch = PolygonIntersector(top_coords, bottom_coords)
    patch.plot()