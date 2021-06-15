import cv2
import numpy as np


class Ball:
	def __init__(self, pos, vel, radius, bounds):
		self.pos = pos
		self.vel = vel
		self.radius = radius
		self.bounds = bounds

	def draw(self, img):
		# Draw the ball on the given image
		pos_int = [int(x) for x in self.pos]
		cv2.circle(img, pos_int, self.radius, (255, 150, 50), -1)

	def update(self, face_pos, face_radius):
		# Update position
		self.pos += self.vel

		# Check for collision with face
		self.check_collision(face_pos, face_radius)

		# Check for collision with boundaries and return game_over flag
		return self.check_bounds()

	def check_bounds(self):
		# Check if ball is touching bottom edge
		if self.pos[1] + self.radius >= self.bounds[1]:
			self.pos[1] = self.bounds[1] - self.radius
			# If yes, game over so return True
			return True

		# Check if ball is touching top edge, rebound if yes
		if self.pos[1] - self.radius <= 0:
			self.pos[1] = self.radius
			self.vel[1] *= -1

		# Check if ball is touching left or right edges, rebound if yes
		if self.pos[0] - self.radius <= 0:
			self.pos[0] = self.radius
			self.vel[0] *= -1
		elif self.pos[0] + self.radius >= self.bounds[0]:
			self.pos[0] = self.bounds[0] - self.radius
			self.vel[0] *= -1

		# Game not over, return False
		return False

	def check_collision(self, face_pos, face_radius):
		# Get squared distance between face and ball
		d_vec = self.pos - face_pos
		dist_sq = np.sum(np.square(d_vec))

		# If this is less than the square of the sum of the radii, collision
		radii_sum = face_radius + self.radius
		if dist_sq <= radii_sum ** 2:
			# Update velocity assuming elastic collision with face at rest
			self.vel -= 2 * np.dot(self.vel, d_vec) / dist_sq * d_vec
			self.pos = d_vec * (radii_sum + 1) / np.sqrt(dist_sq) + face_pos
