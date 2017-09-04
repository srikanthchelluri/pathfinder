import cv2
import numpy as np
from urllib.request import urlopen

from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder


BLUR_PARAM = 100 # Higher means more blurry and darker
MEAN_PARAM = 19 # Higher means more prominent lines
SIGMA_PARAM = 0.1 # Higher means wider threshold (more edges)

SQUARE_PARAM = 400 # Height and width of environment
GRID_PARAM = 8 # Numer of squares per row or per column
POINT_PARAM = 50 # Height and width of point boxes
OBSTACLE_PARAM = 140 # Threshold to detect obstacle, TODO: must adjust


# Given an image, uses computer vision to determine obstacle matrix
def determine_obstacle_matrix(image_url):
	# Load image in grayscale
	original = get_image_gray(image_url)

	# Blur to smooth out lines
	kernel = np.ones((5, 5), np.float32) / BLUR_PARAM
	blur = cv2.filter2D(original, -1, kernel)

	# Brighten due to blur?
	brighten = blur

	# Threshold using adaptive mean
	mean = cv2.adaptiveThreshold(brighten, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, MEAN_PARAM, 2)

	# Invert image
	invert = cv2.bitwise_not(mean)

	# Edge detection
	v = np.median(invert)
	lower = int(max(0, (1.0 - SIGMA_PARAM) * v))
	upper = int(min(255, (1.0 + SIGMA_PARAM) * v))
	edges = cv2.Canny(invert, lower, upper)

	# Isolate largest contour
	image, contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
	board = None
	for c in contours:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		if len(approx) == 4:
			board = approx
			break
	isolated = cv2.drawContours(original.copy(), [board], -1, (0, 255, 0), 3)

	# Perspective transform
	start = np.float32([board[1][0], board[0][0], board[2][0], board[3][0]])
	end = np.float32([[0,0],[SQUARE_PARAM,0],[0,SQUARE_PARAM],[SQUARE_PARAM,SQUARE_PARAM]])
	transform = cv2.getPerspectiveTransform(start, end)
	warp = cv2.warpPerspective(original.copy(), transform, (SQUARE_PARAM, SQUARE_PARAM))

	# Slice image into matrix
	square_size = SQUARE_PARAM / GRID_PARAM
	start_square = square_size / 2
	points = [] # Tuple of all points centered at matrix squares
	for i in range(0, GRID_PARAM):
		row = start_square + (i * square_size)
		for j in range(0, GRID_PARAM):
			col = start_square + (j * square_size)
			points.append((row, col))

	point_squares = []
	for point in points:
		point_squares.append(calculate_square(point))

	contours = warp.copy()
	for ps in point_squares:
		draw = ps.reshape((-1, 1, 2))
		contours = cv2.polylines(contours, [draw], True, (0,255,255))

	# Iterate over matrix boxes and threshold
	matrix = []
	for i in range(0, GRID_PARAM):
		matrix.append([])
		for j in range(0, GRID_PARAM):
			obstacle = calculate_obstacle(warp.copy(), point_squares[i * GRID_PARAM + j])
			matrix[i].append(obstacle)

	# Determine obstacle matrix
	obstacle_matrix = matrix.copy()
	for i in range(0, GRID_PARAM):
		for j in range(0, GRID_PARAM):
			if (obstacle_matrix[i][j] > OBSTACLE_PARAM):
				obstacle_matrix[i][j] = 1
			else:
				obstacle_matrix[i][j] = 0

	# Return matrix
	return obstacle_matrix


# Given an obstacle matrix, uses A* to find a path (represented by points)
def find_path(matrix):
	grid = Grid(matrix=matrix)
	start = grid.node(0, 0) # Given in (x, y) or (col, row)
	end = grid.node(3, 6) # TODO: must change this
	finder = AStarFinder()
	path, runs = finder.find_path(start, end, grid)
	return path


# Given a path represented by points, creates a vector representation
# Go straight: 0, turn left: -1, turn right: 1
def create_vector(path):
	vector = []
	prev = None
	current_direction = 90 # Assume start point is facing south

	for point in path:
		# Omit first point
		if prev == None:
			prev = point
			continue

		# Determine which direction we need to face
		required_direction = None
		diff = (point[0] - prev[0], point[1] - prev[1])
		if diff[0] == 1:
			required_direction = 0
		elif diff[0] == -1:
			required_direction = 180
		elif diff[1] == 1:
			required_direction = 90
		elif diff[1] == -1:
			required_direction = 270
		else:
			return
		prev = point

		# Optimally turn to the direction
		diff_direction = required_direction - current_direction
		if diff_direction == -270 or diff_direction == 90: # Turn right
			vector.append(1)
			vector.append(0)
		elif diff_direction == -180 or diff_direction == 180: # Turn around
			vector.append(1)
			vector.append(1)
			vector.append(0)
		elif diff_direction == -90 or diff_direction == 270: # Turn left
			vector.append(-1)
			vector.append(0)
		elif diff_direction == 0:
			vector.append(0)
		else:
			return
		current_direction = required_direction

	return vector


# HELPER METHODS
def get_image_gray(url):
	resp = urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
	return image

def calculate_square(point):
	delta = POINT_PARAM / 2
	row = point[0]
	col = point[1]

	tl_x = col - delta
	tl_y = row - delta
	tr_x = col + delta
	tr_y = row - delta
	bl_x = col - delta
	bl_y = row + delta
	br_x = col + delta
	br_y = row + delta

	return np.int32([
		[tl_x, tl_y],
		[tr_x, tr_y],
		[br_x, br_y],
		[bl_x, bl_y]
	])

def calculate_obstacle(warp, nparray):
	x = nparray[0][0]
	y = nparray[0][1]
	crop = warp[y:y+POINT_PARAM, x:x+POINT_PARAM]
	return crop.mean()
