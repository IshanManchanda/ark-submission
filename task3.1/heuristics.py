from abc import ABC, abstractmethod

import numpy as np


class HeuristicAbstractBaseClass(ABC):
	# Interface that all heuristic functions need to implement
	@staticmethod
	@abstractmethod
	def evaluate(current, end):
		pass


class HeuristicZero(HeuristicAbstractBaseClass):
	@staticmethod
	def evaluate(current, end):
		return 0


class HeuristicDiagonal(HeuristicAbstractBaseClass):
	@staticmethod
	def evaluate(current, end):
		return np.max(np.abs(end - current))


class HeuristicInfiniteDiagonal(HeuristicAbstractBaseClass):
	@staticmethod
	def evaluate(current, end):
		return 180 * 457 * np.max(np.abs(end - current))


class HeuristicManhattan(HeuristicAbstractBaseClass):
	@staticmethod
	def evaluate(current, end):
		return np.sum(np.abs(end - current))


class HeuristicInfiniteManhattan(HeuristicAbstractBaseClass):
	@staticmethod
	def evaluate(current, end):
		return 180 * 457 * np.sum(np.abs(end - current))
