import os

import cv2
import numpy as np

from ball import Ball

# Load Caffe model
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(
	base_path, "models/res10_300x300_ssd_iter_140000.caffemodel"
)
config_path = os.path.join(base_path, "models/deploy.prototxt.txt")
net = cv2.dnn.readNetFromCaffe(config_path, model_path)


def find_face(img):
	# Convert image to blob for NN input
	# BLOB stands for Binary Large OBject, and is essentially a contour
	# which is represented by certain features like Circularity, Area,
	# Compactness, Perimeter, Convexity, Center of Mass, etc.
	# Args: Img, Scale factor (to scale img), size (of output image),
	# mean (to subtract from img), swapRB (RGB -> BGR), crop
	blob = cv2.dnn.blobFromImage(
		img, 1.0, (300, 300), [104, 117, 123], False, False
	)

	# Set the input to the NN and feedforward
	net.setInput(blob)
	detections = net.forward()  # Returns a list of 4D output blobs

	# return False if no detections
	if not detections.shape[2]:
		return False, None

	# We pick the detection with maximum confidence
	max_confidence = 0
	x1 = y1 = x2 = y2 = 0
	img_h, img_w, _ = img.shape
	for i in range(detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > max_confidence:
			max_confidence = confidence
			x1 = int(detections[0, 0, i, 3] * img_w)
			y1 = int(detections[0, 0, i, 4] * img_h)
			x2 = int(detections[0, 0, i, 5] * img_w)
			y2 = int(detections[0, 0, i, 6] * img_h)

	# If we're less than 50% sure, we report negative
	if max_confidence < 0.5:
		return False, None
	return True, (x1, y1, x2 - x1, y2 - y1)


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
