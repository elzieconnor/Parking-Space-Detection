import numpy as np
import cv2 as cv
import os
import yaml
import random
import requests
import json

confThreshold = 0.3
maskThreshold = 0.3

image = cv.imread('cars.jpg')


# Load names of classes
classesFile = "mscoco_labels.names";
classes = None
with open(classesFile, 'rt') as f:
   classes = f.readlines()

# Load the colors
colorsFile = "colors.txt";
with open(colorsFile, 'rt') as f:
	colorsStr = f.readlines()
colors = []
for i in range(len(colorsStr)):
	rgb = colorsStr[i].split(', ')
	color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
	colors.append(color)

winName = 'Detected Car'
cv.namedWindow(winName, cv.WINDOW_NORMAL)


# Load the network
net = cv.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

#Getting the inputs
outputFile = "maskedImage.jpg"

#cap = cv.VideoCapture(0)
cap = cv.VideoCapture('parkinglot3.jpg')

# Get the video writer initialized to save the output video
if (not image.any()):
	vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 28, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

# Draw the predicted bounding box, colorize and show the mask on the image
def drawBox(frame2, classId, conf, left, top, right, bottom, classMask):
	# Draw a bounding box.
	cv.rectangle(frame2, (left, top), (right, bottom), (255, 178, 50), 3)
	 
	# Resize the mask, threshold, color and apply it on the image
	classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
	mask = (classMask > maskThreshold)
	roi = frame2[top:bottom+1, left:right+1][mask]
 
	#color = colors[classId%len(colors)]
	#Comment the above line and uncomment the two lines below to generate different instance colors
	colorIndex = random.randint(0, len(colors)-1)
	color = colors[colorIndex]
 
	frame2[top:bottom+1, left:right+1][mask] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7 * roi).astype(np.uint8)
 
	# Draw the contours on the image
	mask = mask.astype(np.uint8)
	retr_tree = cv.RETR_TREE
	chain_approx_simp = cv.CHAIN_APPROX_SIMPLE
	im2, contours, hierarchy = cv.findContours(mask, retr_tree, chain_approx_simp)
	cv.drawContours(frame2[top:bottom+1, left:right+1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)

# For each frame, extract the bounding box and mask for each detected object
def postprocess(boxes, masks):
	# Output size of masks is NxCxHxW where
	# N - number of detected boxes
	# C - number of classes (excluding background)
	# HxW - segmentation shape
	numClasses = masks.shape[1]
	numDetections = boxes.shape[2]
	 
	frameH = frame2.shape[0]
	frameW = frame2.shape[1]
	if os.path.exists('coors.yml'):
		os.remove('coors.yml')
	for i in range(numDetections):
		box = boxes[0, 0, i]
		mask = masks[i]
		score = box[2]
		if score > confThreshold:
			classId = int(box[1])
			 
			# Extract the bounding box
			left = int(frameW * box[3])
			top = int(frameH * box[4])
			right = int(frameW * box[5])
			bottom = int(frameH * box[6])
			 
			left = max(0, min(left, frameW - 1))
			top = max(0, min(top, frameH - 1))
			right = max(0, min(right, frameW - 1))
			bottom = max(0, min(bottom, frameH - 1))
			# Extract the mask for the object
			classMask = mask[classId]
			 
			# Draw bounding box, colorize and show the mask on the image
			drawBox(frame2, classId, score, left, top, right, bottom, classMask)
			info = {'coors': []}
			coor = []
			data = []
			carId = i+1
			coor.append((top, left))
			coor.append((bottom, right))
			corner_1 = list(coor[0])
			corner_2 = list(coor[1])
			info['carId'] = carId
			info['coors'] = [corner_1, corner_2]
			data.append(info)
			
			with open('coors.yml','a') as file:
				yaml.dump(data, file)


while True:
	 
	# Get frame from the video
	#hasFrame, frame = cap.read()
	
	r = requests.get('http://127.0.0.1:8080/get_frame',headers = {'cam_id': 'id1'})
	r2 = requests.get('http://127.0.0.1:8080/get_frame',headers = {'cam_id': 'id2'})

	data = r.content
	frame = json.loads(data.decode("utf8"))
	frame = np.asarray(frame, np.uint8)
	frame = cv.imdecode(frame, cv.IMREAD_COLOR)

	data2 = r2.content
	frame2 = json.loads(data2.decode("utf8"))
	frame2 = np.asarray(frame2, np.uint8)
	frame2 = cv.imdecode(frame2, cv.IMREAD_COLOR)
	 
	# Stop the program if reached end of video
	if cv.waitKey(1) == ord('q'):
		#print("Done processing !!!")
		#print("Output file is stored as ", outputFile)
		#cv.waitKey(3000)
		break
 
	# Create a 4D blob from a frame.
	blob = cv.dnn.blobFromImage(frame2, swapRB=True, crop=False)
 
	# Set the input to the network
	net.setInput(blob)
 
	# Run the forward pass to get output from the output layers
	boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
 
	# Extract the bounding box and mask for each of the detected objects
	postprocess(boxes, masks)
 
	# Put efficiency information.
	t, _ = net.getPerfProfile()
	label = 'Mask-RCNN : Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
	cv.putText(frame2, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


	# Write the frame with the detection boxes
	if (image.any()):
		cv.imwrite(outputFile, frame2.astype(np.uint8))
	else:
		vid_writer.write(frame2.astype(np.uint8))
 
	cv.imshow(winName, frame2)
