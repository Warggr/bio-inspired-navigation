''' This code has been adapted from:
***************************************************************************************
*    Title: "Scaling Local Control to Large Scale Topological Navigation"
*    Author: "Xiangyun Meng, Nathan Ratliff, Yu Xiang and Dieter Fox"
*    Date: 2020
*    Availability: https://github.com/xymeng/rmp_nav
*
***************************************************************************************
'''

"""
Getting the Meng 2020 path planner to work
1. Get a topview image of your maze
2. Transform that image into a black(occupied) and white(free space) image
3. Adjust the dimension and origin(i.e. coordinates of the lower left corner)
   of the maze
4. Adjust path_map_dilation and path_map_division until the resulting path map
   looks reasonable (everything in yellow will be considered as occupied space,
                     no waypoints will be generated there)
5. Test by generating a path through the maze, make sure the maze is oriented
   correctly
6. Use the waypoints for whatever you need
"""

import cv2
import math
import numpy as np
import time
import sys
import os

import range_libc

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    from map_occupancy_helpers.map_utils import a_star#, rasterize_line
    from map_occupancy_helpers.math_utils import depth_to_xy_plane, depth_to_xy, compute_normals
    import map_occupancy_helpers.map_utils_cpp as map_cpp
else:
    from .map_occupancy_helpers.map_utils import a_star#, rasterize_line
    from .map_occupancy_helpers.math_utils import depth_to_xy_plane, depth_to_xy, compute_normals
    from .map_occupancy_helpers import map_utils_cpp as map_cpp

from system.types import Vector2D, AllowedMapName


class Map:
    def __init__(self,
                 occupancy_grid,
                 resolution,
                 origin : Vector2D,
                 path_map=None,
                 path_map_division=7,
                 path_map_dilation=2,
                 path_map_weighted_dilation=False,
                 reachable_area_dilation=3,
                 name=None):
        """
        path_map encodes the allowed space for path planning
        reachable_area is path_map shrinked by reachable_area_dilation. This is useful for selecting
        starting point and goal points, which should be on the path map but also should not be too
        close to obstacles.

        :param occupancy_grid: a grayscale image where 255 means free space.
        :param resolution: real width of a pixel in the occupancy map in meters.
        :param destination_map: a colored path map where each color represents a destination.
        :param path_map_dilation:  higher value will shrink the path map further
        :param path_map_weighted_dilation: create the path map from a weighted combination between
                the original path map and the dilated path map. This prevents tight openings from
                being closed due to the dilation process and also allows the agent to start at
                positions anywhere from the original path maps.
        :param reachable_area_dilation:  higher value will shrink the reachable area further
        """

        self.g = {
            'path_map_dilation': path_map_dilation,
            'reachable_area_dilation': reachable_area_dilation,
        }

        self.occupancy_grid = occupancy_grid
        self.occupancy_grid_copy = np.array(self.occupancy_grid, copy=True)

        self.resolution = resolution
        self.origin = origin
        self.name = name

        free_space = (self.occupancy_grid >= 254).astype(np.uint8)
        # hard_obstacles = (self.occupancy_grid == 0).astype(np.uint8)
        self.binary_occupancy = (1 - free_space) * 255  # 0 means no obstacle

        # Compute distance transform of the map
        start_time = time.time()
        self.map_dist_transform, _ = cv2.distanceTransformWithLabels(
            255 - self.binary_occupancy, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # print('distance transform time: %.2f sec' % (time.time() - start_time))

        self.visible_map_bbox = self._compute_visible_map_bbox(True)

        self.path_map_division = path_map_division
        if path_map is None:
            # Automatically generate path map. This is usually the case if we use a real
            # map generated by laser scan.
            path_map_scale = self.resolution / (1.0 / self.path_map_division)

            m = cv2.resize(self.binary_occupancy,
                           dsize=(0, 0), fx=path_map_scale, fy=path_map_scale,
                           interpolation=cv2.INTER_NEAREST)
            self.path_map = m
        else:
            self.path_map = (path_map > 0).astype(np.uint8) * 255

        # Shrink the path map (i.e., dilate the occupied area)
        self.path_map_dilated = self.dilate(self.path_map, path_map_dilation)

        self.reachable_area = 255 - self.dilate(self.path_map_dilated, reachable_area_dilation)
        self.reachable_area_dilation = reachable_area_dilation
        self.reachable_locs = list(zip(*np.nonzero(self.reachable_area)[::-1]))
        self.reachable_locs_per_destination = {}

        if path_map_weighted_dilation:
            self.path_map = (self.path_map * 0.2 + self.path_map_dilated * 0.8).astype(np.uint8)
        else:
            self.path_map = self.path_map_dilated

        self.path_map_dilation = path_map_dilation

        # self.path_map = path_map_dilated

        # import matplotlib.pyplot as plt
        # plt.imshow(self.path_map)
        # plt.show()

        # pad occupancy map to make it a square because it seems range_libc will crash
        # on non-square map.
        h, w = self.binary_occupancy.shape
        self.square_omap = np.zeros((max(h, w), max(h, w)), np.uint8)
        self.square_omap[:w, :h] = np.transpose(self.binary_occupancy, (1, 0))
        self.square_omap_copy = np.array(self.square_omap, copy=True)

        self.omap = range_libc.PyOMap(self.square_omap)

        self.range_scanner = range_libc.PyBresenhamsLine(self.omap, 1000)

    @property
    def shape(self) -> tuple[float, float]:
        return self.binary_occupancy.shape

    def _compute_visible_map_bbox(self, background_traversable):
        if background_traversable:
            nz_indices = np.transpose(np.nonzero(self.occupancy_grid == 0))
        else:
            nz_indices = np.transpose(np.nonzero(self.occupancy_grid != 0))

        # Note that axis 0 is y and axis 1 is x
        # bbox is (x_min, x_max, y_min, y_max)
        grid_x_min, grid_x_max, grid_y_min, grid_y_max = (
            np.min(nz_indices[:, 1]), np.max(nz_indices[:, 1]),
            np.min(nz_indices[:, 0]), np.max(nz_indices[:, 0]))

        division = int(1.0 / self.resolution)
        x_min, y_min = self.continuous_coord(grid_x_min, grid_y_min, division)
        x_max, y_max = self.continuous_coord(grid_x_max, grid_y_max, division)

        return x_min, x_max, y_min, y_max

    def _compute_path_map_area(self):
        """
        :return: the area of path map in meter sq.
        """
        n = np.prod(self.path_map.shape[:2]) - np.count_nonzero(self.path_map)
        return self.resolution ** 2 * n

    def _gen_per_goal_reachable_locations(self):
        free_space = (0, 0, 0)
        opaque_space = (255, 255, 255)
        for x, y in self.reachable_locs:
            # Need to cast to int. Otherwise there will be weird type mismatch errors in opencv.
            r, g, b = (int(_) for _ in self.destination_map[y, x])
            if (r, g, b) != free_space and (r, g, b) != opaque_space:
                if (r, g, b) in self.reachable_locs_per_destination:
                    self.reachable_locs_per_destination[(r, g, b)].append((x, y))
                else:
                    self.reachable_locs_per_destination[(r, g, b)] = []

    def get_reachable_locations(self):
        """
        :return: the reachable locations in the path map's coordinates
        """
        return self.reachable_locs

    def suitable_position_for_robot(self, p : Vector2D) -> bool:
        # print(p, 'is suitable:', end='')
        map_coords = self.map_coord_to_path_coord(p[0], p[1])
        try:
            return self.path_map[map_coords[::-1]] <= 128
        except IndexError:
            return False

    def grid_coord(self, x, y, n_division):
        '''
        :return: the grid coordinates where (x, y) falls
        '''
        return int((x - self.origin[0]) * n_division), int((y - self.origin[1]) * n_division)

    def grid_coord_batch(self, xys, n_division):
        """
        Batch version of grid_coord()
        :param xys: N x 2 np array
        """
        return ((xys - np.array(self.origin)) * n_division).astype(np.int32)

    def continuous_coord(self, x, y, n_division) -> Vector2D:
        """
        The inverse operation of grid_coord()
        """
        return float(x) / n_division + self.origin[0], float(y) / n_division + self.origin[1]

    def map_coord_to_occupancy_grid_coord(self, x, y):
        return self.grid_coord(x, y, int(1.0 / self.resolution))

    def path_coord_to_map_coord(self, x, y):
        return self.continuous_coord(x, y, self.path_map_division)

    def map_coord_to_path_coord(self, x, y):
        return self.grid_coord(x, y, self.path_map_division)

    def find_reachable_area(self, a, loc):
        from collections import deque
        res = np.zeros(a.shape, np.uint8).tolist()
        h, w = a.shape

        q = deque([loc])
        res[loc[1]][loc[0]] = 255

        while len(q) > 0:
            x, y = q.popleft()
            assert res[y][x] == 255, res[y][x]
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    y2 = y + i
                    x2 = x + j
                    if 0 <= x2 < w and 0 <= y2 < h and res[y2][x2] == 0 and a[y2, x2] == 0:
                        res[y2][x2] = 255
                        q.append((x2, y2))

        return np.array(res, np.uint8)

    def dilate(self, src, n_iter):
        import cv2
        distance_map = cv2.distanceTransform(np.uint8(255.0 - src), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dilated = np.clip(255.0 - (distance_map / n_iter) * 255.0, 0.0, None)
        return dilated

    def find_path(self, start_pos: Vector2D, goal_pos: Vector2D) -> list[Vector2D]:
        assert self.suitable_position_for_robot(start_pos) and self.suitable_position_for_robot(goal_pos)
        start_coord = self.grid_coord(start_pos[0], start_pos[1], self.path_map_division)
        goal_coord = self.grid_coord(goal_pos[0], goal_pos[1], self.path_map_division)

        waypoints = a_star(self.path_map, start_coord, goal_coord, soft_obstacle_scale=1.0)
        assert waypoints[-1] == tuple(goal_coord) and waypoints[0] == tuple(start_coord), (waypoints[0], waypoints[-1], start_coord, goal_coord)

        # if waypoints is None:
        #    return [(start_pos, goal_pos)]

        from numbers import Number
        res = []
        for x, y in waypoints:
            cx, cy = self.continuous_coord(x, y, self.path_map_division)
            assert isinstance(cx, Number)
            res.append((cx, cy))

        return res

    def find_path_destination(self, start_pos, dest):
        """
        Return a path from start_pos to the specified destination. This is fast because all
        possible paths have been precomputed.
        """
        paths = self.destination_paths[dest]
        cx, cy = self.destination_centroids[dest]

        sx, sy = self.grid_coord(start_pos[0], start_pos[1], self.path_map_division)
        if (sx, sy) not in paths:
            print('find_path_destination returns None. dest', dest, 'cx', cx, 'cy', cy, 'sx', sx, 'sy', sy)
            print('path_map[sy, sx] =', self.path_map[sy, sx])
            print('destination_map[sy, sx] =', self.destination_map[sy, sx])
            return None

        path = []

        x, y = sx, sy
        while (x, y) != (cx, cy):
            path.append(tuple(self.path_coord_to_map_coord(x, y)))
            x, y = paths[(x, y)]

        path.append(tuple(self.path_coord_to_map_coord(cx, cy)))

        return path

    def no_touch(self, x1, y1, x2, y2, tolerance=0.0):
        """
        Check whether the line (x1, y1) (x2, y2) touches any obstacle with specified tolerance.
        Positive tolerance means that we allow the line to intersect with obstacle more.
        Negative tolerance means that we consider touch when the line is close to obstacles.
        :return: True if no touch.
        """
        queries = np.zeros((1, 3), dtype=np.float32)
        result = np.zeros(1, dtype=np.float32)

        x, y = self.grid_coord(x1, y1, int(1.0 / self.resolution))
        queries[:, 0] = x
        queries[:, 1] = y
        queries[0, 2] = np.arctan2(y2 - y1, x2 - x1)

        self.range_scanner.calc_range_many(queries, result)
        result *= self.resolution
        dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return dist - tolerance < result[0]

    def no_touch_batch(self, lines, tolerance=0.0):
        """
        Batch version of no_touch()
        :param lines: N x 4 np array. Each row is one line (x1, y1, x2, y2)
        :param tolerance: vector of size N.
        :return: boolean array of size N.
        """
        n = len(lines)

        queries = np.zeros((n, 3), dtype=np.float32)
        result = np.zeros(n, dtype=np.float32)

        xy1s = self.grid_coord_batch(lines[:, :2], int(1.0 / self.resolution))
        xy2s = self.grid_coord_batch(lines[:, 2:], int(1.0 / self.resolution))

        queries[:, :2] = xy1s
        queries[:, 2] = np.arctan2(xy2s[:, 1] - xy1s[:, 1], xy2s[:, 0] - xy1s[:, 0])
        self.range_scanner.calc_range_many(queries, result)

        result *= self.resolution
        dists = np.linalg.norm(lines[:, :2] - lines[:, 2:], axis=1)
        return dists - tolerance < result

    def visible(self, x1, y1, x2, y2, distance_thres=0.0):
        xx1, yy1 = self.grid_coord(x1, y1, int(1.0 / self.resolution))
        xx2, yy2 = self.grid_coord(x2, y2, int(1.0 / self.resolution))
        dist_thres_grid = int(distance_thres / self.resolution)

        # TODO uncomment when have dylib
        return map_cpp.visible(self.map_dist_transform, xx1, yy1, xx2, yy2, dist_thres_grid)

        # Equivalent to map_utils_cpp.visible(), but a lot slower.

        # h, w = self.map_dist_transform.shape
        # points = rasterize_line(xx1, yy1, xx2, yy2)
        # for x, y in points:
        #     if 0 <= x < w and 0 <= y < h and self.map_dist_transform[y, x] > dist_thres_grid:
        #         continue
        #     else:
        #         return False
        #
        # return True

    def get_1d_depth(self, pos, n_depth_ray, heading=0.0, fov=np.pi * 2.0):
        '''
        Get depth measurement at location x, y with heading and fov. The depth rays are from right
        to left.
        '''

        queries = np.zeros((n_depth_ray, 3), dtype=np.float32)
        result = np.zeros(n_depth_ray, dtype=np.float32)

        x, y = self.grid_coord(pos[0], pos[1], int(1.0 / self.resolution))
        queries[:, 0] = x
        queries[:, 1] = y

        for i in range(n_depth_ray):
            # FIXME: hack
            if abs(fov - np.pi * 2.0) < 1e-5:
                theta = float(i) / n_depth_ray * fov
            else:
                theta = float(i) / n_depth_ray * fov - fov * 0.5

            queries[i, 2] = (theta + heading)

        self.range_scanner.calc_range_many(queries, result)

        result *= self.resolution

        return result

    def get_1d_depth_plane(self, pos, n_depth_ray, heading=0.0, fov=np.pi * 2.0):
        '''
        Simulate depth projection on an image plane. You can think of it as a 1d depth camera.
        Different from a laser scanner, angles between depths ray are changing.
        Get depth measurement at location x, y with heading and fov. The depth rays are from right
        to left.
        '''

        queries = np.zeros((n_depth_ray, 3), dtype=np.float32)
        result = np.zeros(n_depth_ray, dtype=np.float32)

        x, y = self.grid_coord(pos[0], pos[1], int(1.0 / self.resolution))
        queries[:, 0] = x
        queries[:, 1] = y

        assert fov < np.pi

        w = np.tan(fov / 2) * 2.0

        for i in range(n_depth_ray):
            x = w / 2 - i * w / n_depth_ray
            theta = np.arctan2(1.0, x) - np.pi / 2
            queries[i, 2] = (theta + heading)

        self.range_scanner.calc_range_many(queries, result)

        result *= self.resolution

        assert not np.all(result == 0)

        return result

    def view_overlap(self, pos1, heading1, pos2, heading2, fov,
                     n_test_rays=100, offset=0.06, mode='plane', vis=None):
        """
        Estimate the overlapping area between two camera poses
        :param offset: the amount to offset laser points to make them outside obstacles. The default
                       value should work in most cases.
               mode: 'plane' if assuming a depth camera. 'lidar' if assuming a lidar.
        :return: (percentage of the image area in camera1 that is visible in camera2,
                  percentage of the image area in camera2 that is visible in camera1)
        """

        # Shrink depths slightly to reduce the possibility of laser points being trapped into
        # obstacles.
        def offset_points(xy):
            normals = compute_normals(xy)
            assert not np.isnan(normals[0, 0]), xy
            return xy + normals * offset

        assert fov != 0

        if mode == 'plane':
            depth1 = self.get_1d_depth_plane(pos1, n_test_rays, heading1, fov)
            depth2 = self.get_1d_depth_plane(pos2, n_test_rays, heading2, fov)
            xy1 = offset_points(depth_to_xy_plane(depth1, pos1, heading1, fov))
            xy2 = offset_points(depth_to_xy_plane(depth2, pos2, heading2, fov))
        elif mode == 'lidar':
            depth1 = self.get_1d_depth(pos1, n_test_rays, heading1, fov)
            depth2 = self.get_1d_depth(pos2, n_test_rays, heading2, fov)
            xy1 = offset_points(depth_to_xy(depth1, pos1, heading1, fov))
            xy2 = offset_points(depth_to_xy(depth2, pos2, heading2, fov))
        else:
            raise RuntimeError('Unsupported mode %s' % mode)

        # Non vectorized version
        # def inside_fov(x, y, pos, heading, fov):
        #     """
        #     :return: True if (x, y) is inside the camera defined by (pos, heading, fov)
        #     """
        #     x1, y1 = x - pos[0], y - pos[1]
        #     norm = math.sqrt(x1 ** 2 + y1 ** 2) + 1e-5
        #     x1, y1 = x1 / norm, y1 / norm
        #     ux, uy = math.cos(heading), math.sin(heading)
        #     return x1 * ux + y1 * uy > math.cos(fov * 0.5)
        # def visible(x, y, pos, heading, fov):
        #     return inside_fov(x, y, pos, heading, fov) and self.visible(x, y, pos[0], pos[1])

        def inside_fov(xy, pos, heading, fov):
            xy2 = xy - np.array(pos, np.float32)
            norm = np.linalg.norm(xy2, axis=1, ord=2, keepdims=True) + 1e-5
            xy_normed = xy2 / norm
            ux, uy = math.cos(heading), math.sin(heading)
            return np.dot(xy_normed, (ux, uy)) > math.cos(fov * 0.5)

        def visible(xy, pos, heading, fov):
            inside_fov_mask = inside_fov(xy, pos, heading, fov).tolist()
            visible_mask = [inside_fov_mask[i] & self.visible(x, y, pos[0], pos[1])
                            for i, (x, y) in enumerate(xy.tolist())]
            return visible_mask

        visible_in_cam1 = visible(xy2, pos1, heading1, fov)
        visible_in_cam2 = visible(xy1, pos2, heading2, fov)

        # xs, ys = zip(*xy1)
        # vis.scatter('depth1', xs, ys, c='r', marker='+')
        # xs, ys = zip(*xy2)
        # vis.scatter('depth2', xs, ys, c='g', marker='+')
        #
        # if len(visible_in_cam2) > 0:
        #     xs, ys = zip(*[xy1[i] for i in range(n_test_rays) if visible_in_cam2[i]])
        #     vis.scatter('visible_in_cam2', xs, ys, c='r', marker='x')

        # if len(visible_in_cam1) > 0:
        #     xs, ys = zip(*visible_in_cam1)
        #     vis.scatter('visible_in_cam1', xs, ys, c='g', marker='x')

        return sum(visible_in_cam2) / float(n_test_rays), sum(visible_in_cam1) / float(n_test_rays)

    def view_overlap_matrix(self, locations, headings, fov, n_test_rays=100, offset=0.06):
        """
        Same as view_overlap(), but compute pairwise overlaps and returns a matrix
        A matrix of size N x N where (i, j) is the overlap between camera i and j
        """

        assert len(locations) == len(headings)

        def offset_points(xy):
            normals = compute_normals(xy)
            return xy + normals * offset

        depths = [self.get_1d_depth_plane(loc, n_test_rays, heading, fov)
                  for loc, heading in zip(locations, headings)]

        xys = [offset_points(depth_to_xy_plane(depth, loc, heading, fov))
               for depth, loc, heading in zip(depths, locations, headings)]

        def inside_fov(xy, pos, heading, fov):
            xy2 = xy - np.array(pos, np.float32)
            norm = np.linalg.norm(xy2, axis=1, ord=2, keepdims=True) + 1e-5
            xy_normed = xy2 / norm
            ux, uy = math.cos(heading), math.sin(heading)
            return np.dot(xy_normed, (ux, uy)) > math.cos(fov * 0.5)

        def visible(xy, pos, heading, fov):
            inside_fov_mask = inside_fov(xy, pos, heading, fov).tolist()
            visible_mask = [inside_fov_mask[i] & self.visible(x, y, pos[0], pos[1])
                            for i, (x, y) in enumerate(xy.tolist())]
            return visible_mask

        n = len(locations)

        overlaps = np.zeros((n, n), np.float32)

        for i in range(n):
            for j in range(n):
                if i == j:
                    overlaps[i, i] = n_test_rays
                    continue
                overlaps[i, j] = sum(visible(xys[i], locations[j], headings[j], fov))
                overlaps[j, i] = sum(visible(xys[j], locations[i], headings[i], fov))

        overlaps /= n_test_rays

        return overlaps


def environment_dimensions(env_model: AllowedMapName):
    if env_model == "Savinov_val3":
        return np.array([-9, -6]), np.array([6, 4])
    elif env_model == "Savinov_val2":
        return np.array([-5, -5]), np.array([5, 5])
    elif env_model == "Savinov_test7":
        return np.array([-9, -4]), np.array([6, 4])
    elif env_model == "plane":
        return None
    elif "obstacle" in env_model:
        return None
    elif env_model == "linear_sunburst":
        return np.array([0, 0]), np.array([11, 11])
    else:
        raise ValueError("No matching env_model found.")

class MapLayout(Map):
    def __init__(self, name):
        """
        Used to 
            - generate a path from point A to point B through the maze or
            - find the amount of overlap of two different agent povs
        
        arguments:
        name    -- environment name (Savinov_val3, Savinov_val2, Savinov_test7)

        """
        if name.startswith("linear_sunburst."):
            name = "linear_sunburst_map"
        dirname = os.path.dirname(__file__)
        filepath = os.path.realpath(os.path.join(dirname, str(name) + "/maze_topview_binary.png"))
        # print(filepath)
        img = cv2.imread(filepath)
        img = cv2.flip(img, 0)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        '''3. Adjust the dimension and origin(i.e. coordinates of the lower left corner)
           of the maze'''
        origin, corner = environment_dimensions(name)
        dim = corner - origin

        # approximate actual space represented by one pixel
        res = ((dim[0] / len(img[0])) + (dim[1] / len(img))) / 2

        '''4. Adjust path_map_dilation and path_map_division until the resulting path map
           looks reasonable (everything in yellow will be considered as occupied space,
                             no waypoints will be generated there)'''
        super().__init__(img_grey, res, origin, path_map_division=20,
                         path_map_dilation=7, reachable_area_dilation=3,
                         path_map_weighted_dilation=True, name=name)

    ''' Transform PNG of top view of map layout to black(occupied) and white(free) image'''
    ''' 2. Transform that image into a black(occupied) and white(free space) image'''

    def png_to_binary(self, filepath):
        filepath = os.path.join(os.path.dirname(__file__), filepath)
        im_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite(filepath[:-4] + "_binary.png", im_bw)

    '''5. Test by generating a path through the maze, make sure the maze is oriented
       correctly'''

    def draw_map_path(self, start=[-2, 0.3], goal=[-4, 0.5], index=None):
        from system.controller.simulation.environment.map_occupancy_helpers.map_visualizer import OccupancyMapVisualizer
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        im = plt.imshow([[1, 2], [3, 4]])
        fig.canvas.draw()
        ax.draw_artist(im)
        vis = OccupancyMapVisualizer(self, ax)
        vis.draw_map()
        waypoints = self.find_path(start, goal)
        if waypoints:
            x, y = np.array(waypoints).T
            plt.scatter(x, y)
            ax.annotate(index, start)
            plt.show()

    def draw_path(self, path):
        from system.controller.simulation.environment.map_occupancy_helpers.map_visualizer import OccupancyMapVisualizer
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)
        im = plt.imshow([[1, 2], [3, 4]])
        fig.canvas.draw()
        ax.draw_artist(im)
        vis = OccupancyMapVisualizer(self, ax)
        vis.draw_map()
        for i in range(len(path) - 1):
            waypoints = self.find_path(path[i], path[i + 1])
            if waypoints:
                x, y = np.array(waypoints).T
                plt.scatter(x, y, c='#0065BD')
        plt.axis("off")
        plt.show()


import random
from system.controller.simulation.environment.map_occupancy import MapLayout


def random_coordinates(xmin, xmax, ymin, ymax, rng=random.Random()):
    return np.array([ rng.uniform(xmin, xmax), rng.uniform(ymin, ymax) ])

def random_points(env_model: AllowedMapName, rng: random.Random) -> tuple[Vector2D, Vector2D]:
    map = MapLayout(env_model)
    origin, corner = environment_dimensions(env_model)
    dimensions = [ origin[0], corner[0], origin[1], corner[1] ]

    result = []
    for i in range(2):
        i = 0
        while True:
            i += 1
            p = random_coordinates(*dimensions, rng=rng)
            #print(f"[i={i}]Trying", p, "...")
            if map.suitable_position_for_robot(p):
                break
        result.append(p)
    return tuple(result)


if __name__ == "__main__":
    """ To compute the waypoints on the path in a particular maze ...
    ... first create the binary of the maze layout
    ... create the layout in the MapLayout class
    instantiate  and try it out by drawing the path.
    """

    savinov = MapLayout("Savinov_val3")
    savinov.png_to_binary('Savinov_val3/maze_topview.png')
    savinov.draw_map_path()

    savinov7 = MapLayout("Savinov_test7")
    savinov7.png_to_binary('Savinov_test7/maze_topview.png')
    savinov7.draw_map_path()

    savinov2 = MapLayout("Savinov_val2")
    savinov2.png_to_binary('Savinov_val2/maze_topview.png')
    savinov2.draw_map_path()
