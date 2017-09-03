import cv2
import numpy as np

from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

# Given an image, uses computer vision to determine obstacle matrix
def determine_obstacle_matrix(image):
	return None

# Given an obstacle matrix, uses A* to find a path (represented by points)
def find_path(matrix):
	grid = Grid(matrix=matrix)
	start = grid.node(0, 0)
	end = grid.node(7, 7)
	finder = AStarFinder()
	path, runs = finder.find_path(start, end, grid)
	return path

# Given a path represented by points, creates a vector representation
def create_vectors(path):
	return None