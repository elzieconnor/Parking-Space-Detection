import numpy as np
import json
import requests
import cv2

class Camera(object):
	def __init__(self, mode, addr, id):
		self.id = id
		self.video = None
		if(mode == "file" and type(addr) == type("hi")):
			self.video = cv2.VideoCapture(addr)
		elif(mode == "live" and type(addr) == type(0)):
			self.video = cv2.VideoCapture(addr)
		else:
			print("ERROR: Camera class given either an incorrect mode or an address of the wrong type.")

	def __del__(self):
		self.video.release()
	
	def get_cam_id(self):
		return self.id
		
	def get_frame(self):
		ret, frame = self.video.read()
		if(ret == True):
			return frame
		else:
			return None

	#Normalizing image to be 0.6 of the original width and height
	def norm_frame(self, frame):
		frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
		return frame

	#Encoding frame from OpenCV format to JPEG compression format
	def package(self, frame):
		frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tolist()
		frame = json.dumps(frame)
		return frame
	
	def cam_open(self):
		opened = self.video.isOpened()
		return opened
		
