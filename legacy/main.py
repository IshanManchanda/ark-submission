import os

import cv2
import numpy as np

from ball import Ball

# Load Haarcascade file
base_path = os.path.dirname(os.path.abspath(__file__))
face_classifier = cv2.CascadeClassifier(
	os.path.join(base_path, 'models/haarcascade_frontalface.xml')
)


def find_face(img):
	# Convert the img to grayscale and then run haarcascade-based
	# face detection
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiScale(gray, 1.3, 5)

	# No faces found, return False for success
	if len(faces) == 0:
		return False, None

	# Multiple faces found, we'll pick the face with the largest bounding area
	if len(faces) > 1:
		return True, max(faces, key=lambda f: f[2] * f[3])

	# Found a single face, return it along with True for success
	return True, faces[0]


def main():
	# TODO: Look into automatically getting maximum resolution
	# We use the DirectShow backend to use maximum camera resolution
	try:
		cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
		size = np.array([1280, 720])
		cam.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
		cam.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
	except Exception as e:
		print(e)
		print("Backend and camera resolution values may need to be changed.")
		try:
			cam.release()
		except Exception:
			pass
		return

	# Create a named window to hold the game and resize it
	cv2.namedWindow("game", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("game", *size)

	# Create ball object with initial position, velocity, and size
	pos = np.array([100, 100], dtype=np.float64)
	vel = 15 * np.array([2, 2], dtype=np.float64)
	ball = Ball(pos, vel, 60, size)

	# Initialize variables to hold detected face information
	face_x = face_y = face_w = face_h = 0

	while True:
		# Read in frame from camera and terminate if unsuccessful
		ret, frame = cam.read()
		if not ret:
			print("failed to grab frame")
			break

		# Mirror frame horizontally to make game more intuitive
		frame = cv2.flip(frame, 1)

		# Apply face detection and get coordinates + size
		success, face_data = find_face(frame)

		# If no face was detected, we'll simply use the previous frame's
		# coordinates but increase the size slightly to accommodate movement
		if success:
			face_x, face_y, face_w, face_h = face_data
		else:
			face_x -= int(face_w * 0.005)
			face_y -= int(face_h * 0.005)
			face_w = int(face_w * 1.01)
			face_h = int(face_h * 1.01)

		# Get face position and radius for collision detection and masking
		face_pos = np.array([face_x + face_w // 2, face_y + face_h // 2])
		face_radius = max(face_w, face_h) // 2

		# Apply mask to keep only the detected face
		mask = np.zeros(size[::-1], dtype=np.uint8)
		cv2.circle(mask, face_pos, face_radius, 1, -1)
		game_frame = cv2.bitwise_and(frame, frame, mask=mask)

		# Update the ball's position, checking for collisions
		game_over = ball.update(face_pos, face_radius)

		# Draw the ball on the current frame
		ball.draw(game_frame)

		# If game over, show message and wait for user to press key
		if game_over:
			print("Game Over. Press any key to continue")
			cv2.waitKey(0)
			break

		# Draw the frame
		cv2.imshow("game", game_frame)

		# Check if user has pressed ESC and quit if yes
		k = cv2.waitKey(1)
		if k % 256 == 27:
			print("Escape pressed, exiting")
			break

	# Cleanup
	cam.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
