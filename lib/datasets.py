import cv2
import yaml
import numpy as np
import requests
import json
#from camera_client import Camera

file_path = 'storage\config\id1.yml'
r = requests.get('http://127.0.0.1:8080/get_frame', headers= {'cam_id': 'id1'})
data = r.content
frame = json.loads(data.decode("utf8"))
frame = np.asarray(frame, np.uint8)
img = cv2.imdecode(frame, cv2.IMREAD_COLOR)
refPt = []
data = []
i = 1
cropping = False

def yaml_loader(file_path):
	with open(file_path, "r") as file_descr:
		data = yaml.load(file_descr)
		return data

def yaml_dump(file_path, data):
	with open(file_path, "a") as file_descr:
		yaml.dump(data, file_descr)


def click_and_crop(event, x, y, flags, param):
	info = {'id': 0, 'points': []}
	global refPt, cropping, i
	
	if event == cv2.EVENT_LBUTTONDBLCLK:
		refPt.append((x,y))
		cropping = False

	if len(refPt) == 4:

		cv2.line(image, refPt[0], refPt[1], (0, 0, 255), 1)
		cv2.line(image, refPt[1], refPt[2], (0, 0, 255), 1)
		cv2.line(image, refPt[2], refPt[3], (0, 0, 255), 1)
		cv2.line(image, refPt[3], refPt[0], (0, 0, 255), 1)

		corner_1 = list(refPt[2])
		corner_2 = list(refPt[3])
		corner_3 = list(refPt[0])
		corner_4 = list(refPt[1])

		info['points'] = [corner_1, corner_2, corner_3, corner_4]
		info['id'] = i
		data.append(info)
		refPt = []
		i += 1

image = img.copy()
clone = image.copy()
cv2.namedWindow("Click to mark points")
cv2.imshow("Click to mark points", image)
cv2.setMouseCallback("Click to mark points", click_and_crop)

while True:
	cv2.imshow("Click to mark points", image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	   
# data list into yaml file
if data != []:
	yaml_dump(file_path, data)
cv2.destroyAllWindows()