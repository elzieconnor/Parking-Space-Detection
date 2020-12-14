from flask import Flask, render_template, request, Response
from lib.cam import Camera
import json
import numpy as np
import cv2
FEEDS = {} #Format: {"<ID>: Camera Object"}
app = Flask(__name__)

@app.route('/')
def cam_index():
	return render_template('cam_index.html')

@app.route('/get_frame', methods = ['GET'])
def get_frame():
	print(request.headers)

	#STEP-1: Locate specific camera
	cam_id = request.headers['cam_id']

	#STEP-2: Pull-norm-package frame
	frame = FEEDS[cam_id].get_frame()
	print(frame)
	if np.shape(frame) == ():
		#make a blue frame for camera object
		frame = np.zeros([480, 640], dtype=np.uint8)
		frame.fill(100)
		label = str(cam_id) + ' is OFFLINE'
		cv2.putText(frame, label, (240, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
		frame = FEEDS[cam_id].package((FEEDS[cam_id].norm_frame(frame)))
	else:
		frame = FEEDS[cam_id].package((FEEDS[cam_id].norm_frame(frame)))
	
	#STEP-3: return frame
	return frame


def init():
	global FEEDS
	#STEP-01: Build the Feeds Dict from user input
	x = True
	num = 1
	while x:
		i = 'id'
		mode = input('Is the video feed from camera "live" or from a "file": ')
		if mode == 'live':
			addr = input('Enter the address of the live video as a integer: ')
			addr = int(addr)
		else:
			addr = input('Enter the file name of the video: ')
		cam_id = input('Enter in camera ID for video feed: ')
		print(mode)
		print(addr)
		FEEDS[i+str(num)] = Camera(mode, addr, cam_id)
		print(FEEDS)
		ans = input('Enter "q" to quit: ')
		if ans == 'q':
			x = False
		num += 1

	#STEP-02: Check video feeds
	if len(FEEDS) == 2:
		video_feeds = np.hstack((FEEDS['id1'].get_frame(),FEEDS['id2'].get_frame()))
	else:
		video_feeds = FEEDS['id1'].get_frame()
	while(True):
		cv2.imshow("frame", video_feeds)
		#cv2.imshow("frame", FEEDS['id2'].get_frame())
		if cv2.waitKey(1) == ord('q'):
			break
	cv2.destroyAllWindows()

	#STEP-03: return True/False if setup failed
	input('Press Enter to continue...')

if __name__ == '__main__':
	init()
	app.run(host='0.0.0.0', port='8080', debug=False)
