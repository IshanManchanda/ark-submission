import itertools
import time
import os

import cv2
import numpy as np

from heuristics import HeuristicDiagonal, HeuristicInfiniteDiagonal, \
	HeuristicInfiniteManhattan, HeuristicManhattan, HeuristicZero
from pathfinder import PathFinder


def visualize(final_state, i, codes):
	# Function to render the final state of the solved maze

	# Color values
	colors = {
		codes['wall']: (255, 255, 255),
		codes['path']: (0, 0, 0),
		codes['start']: (113, 204, 45),
		codes['end']: (60, 76, 231),
		codes['best_path']: (250, 60, 90),
		codes['explored']: (00, 120, 250),
		codes['frontier']: (255, 150, 50),
	}

	# Generate image by mapping final_state values to corresponding colors
	img = np.array(
		[[colors[x] for x in row] for row in final_state], dtype=np.uint8
	)
	# Scale up image
	out = cv2.resize(img, (2285, 900), interpolation=cv2.INTER_NEAREST)

	# Save image
	base_path = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(base_path, f'out/maze_solved_{i}.png')
	cv2.imwrite(path, out)


def save_data(data, filename):
	# Write data to file
	base_path = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(base_path, f'out/maze_{filename}.csv')
	np.savetxt(path, data, delimiter=', ', fmt='%g')


def main():
	# Read in the maze image
	maze = cv2.imread('data/maze_lv3.png')

	# Use the found color to create a mask
	color = 230
	mask = maze[:, :, 0] != color

	# Use 1 to mark pathways and use 0 to mark obstacles
	maze[mask] = [1, 1, 1]
	maze[~mask] = [0, 0, 0]
	# Convert to 2d
	maze = maze[:, :, 0]

	# Start and end points found from the image
	start = np.array([150, 20])
	end = np.array([160, 430])

	# List of valid moves
	# In the first case we'll consider only up, down, left, and right
	# while in the second we'll allow diagonal moves as well
	moves_4 = np.array([[0, -1], [-1, 0], [0, 1], [1, 0]])
	moves_diagonal = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
	moves_8 = np.append(moves_4, moves_diagonal, axis=0)

	# For both sets of moves, we evaluate 3 variants:
	# Dijkstra, an Admissible A*, and a Best-First Search.
	variants = [
		(moves_4, HeuristicZero),
		(moves_4, HeuristicManhattan),
		(moves_4, HeuristicInfiniteManhattan),
		(moves_8, HeuristicZero),
		(moves_8, HeuristicDiagonal),
		(moves_8, HeuristicInfiniteDiagonal),
	]

	# Lists to hold the values of interest
	path_lengths = []
	explored_nodes = []
	runtimes = []

	# Codes for various elements in the final state,
	# used for convenient mapping of colors for the visualization step.
	codes = {
		'wall': 0,
		'path': 1,
		'start': 2,
		'end': 3,
		'best_path': 4,
		'explored': 5,
		'frontier': 6,
	}

	# Counter generates unique numbers for the figure indices
	c = itertools.count(1)

	# Iterate over all cases we're considering
	for moves, heuristic in variants:
		# Create a pathfinder object
		pathfinder = PathFinder(maze, start, end, moves, heuristic)

		# Solve maze with a timer
		t_start = time.perf_counter()
		path_length = pathfinder.solve()
		t_end = time.perf_counter()

		# Visualize the final state to get the password
		# (Stores the image in the out directory)
		final_state, explored_count = pathfinder.get_final_state(codes)
		visualize(final_state, next(c), codes)

		# Store performance metrics of this run
		path_lengths.append(path_length)
		explored_nodes.append(explored_count)
		runtimes.append(t_end - t_start)

	# Save collected data to files
	save_data(np.array(path_lengths).reshape(2, -1), 'path_lengths')
	save_data(np.array(explored_nodes).reshape(2, -1), 'explored_nodes')
	save_data(np.array(runtimes).reshape(2, -1), 'runtimes')


if __name__ == '__main__':
	main()
