import heapq
import itertools
from collections import deque

import numpy as np

from heuristics import HeuristicAbstractBaseClass


class PathFinder:
	def __init__(self, maze, start, end, moves, heuristic):
		# nxm grid that represents the maze
		self.maze = maze
		self.size = maze.shape

		# Start and End positions
		self.start = start
		self.end = end

		# Array of possible moves
		self.moves = moves

		# Heuristic function, must implement the interface
		assert issubclass(heuristic, HeuristicAbstractBaseClass)
		self.heuristic = heuristic

		# Initialize distance matrix
		self.distances = np.full(self.size, np.inf)
		# self.distances[self.start[0]][self.start[1]] = 0
		self.distances[self.start[0], self.start[1]] = 0

		self.explored = 0
		self.q = []
		self.path = deque()

	def solve(self):
		# Initialize priority queue with start node
		# The queue element is of format: (cost, tiebreaker, node)
		heapq.heappush(self.q, (0, 0, self.start))
		c = itertools.count()
		while self.q:
			# Get the current element to explore
			# dist_current, _, current = heapq.heappop(self.q)
			current = heapq.heappop(self.q)[2]
			g = self.distances[current[0], current[1]] + 1

			for move in self.moves:
				# Get the adjacent point after our move
				adj = current + move

				# If the position is invalid, continue
				if not self.is_position_valid(adj):
					continue

				# If we've already explored this node, continue
				# unless we have found a shorter path than previous.
				# This can cause multiple instances of the same node to be
				# in the queue at the same time, but it doesn't impact us
				# as the neighbors will not be enqueued multiple times.
				if self.distances[adj[0], adj[1]] <= g:
					continue

				# If we've arrived at the target, stop exploring
				if np.all(adj == self.end):
					self.get_shortest_path()
					return len(self.path) - 1

				# Compute cost of node
				h = self.heuristic.evaluate(adj, self.end)
				cost = g + h

				# We use a tiebreaker for when nodes have equal priority
				# This (incrementing) tiebreaker ensures FIFO behavior
				tiebreaker = next(c)

				# Enqueue the node and update its distance
				self.q.append((cost, tiebreaker, adj))
				self.distances[adj[0], adj[1]] = g

	def is_position_valid(self, pos):
		# Check if the new position is out of bounds
		if not np.all(0 < pos) or not np.all(pos < self.size):
			return False

		# Check if the new position is an obstacle
		return self.maze[pos[0], pos[1]]

	def get_shortest_path(self):
		# To get the shortest path, we move backwards from the end node
		current = self.end
		self.path.appendleft(current)

		while not np.all(current == self.start):
			# Get the minimum distance neighbor of the current node
			min_node = None
			min_dist = np.inf
			for move in self.moves:
				new_node = current + move

				# Check for neighbor validity, don't want to index out of bounds
				if not self.is_position_valid(new_node):
					continue

				if self.distances[new_node[0], new_node[1]] < min_dist:
					min_node = new_node
					min_dist = self.distances[min_node[0], min_node[1]]

			# Pick the node with the minimum distance and add to path
			current = min_node
			self.path.appendleft(current)

	def get_final_state(self, codes):
		# Copy the maze to represent the final state for visualization
		# Set all explored nodes (dist != inf) as explored
		final_state = self.maze.copy()
		final_state[self.distances != np.inf] = codes['explored']

		# Set the frontier nodes and find the number of unique frontier nodes
		# to calculate the number of explored nodes
		unique_frontier = 0
		while self.q:
			current = heapq.heappop(self.q)[2]

			if final_state[current[0], current[1]] != codes['frontier']:
				unique_frontier += 1
			final_state[current[0], current[1]] = codes['frontier']

		# Set the best path nodes
		while self.path:
			current = self.path.pop()
			final_state[current[0], current[1]] = codes['best_path']

		# Set the initial and final nodes
		final_state[self.start[0], self.start[1]] = codes['start']
		final_state[self.end[0], self.end[1]] = codes['end']

		# Compute the number of explored nodes
		explored_nodes = np.sum(self.distances != np.inf) - unique_frontier
		return final_state, explored_nodes
