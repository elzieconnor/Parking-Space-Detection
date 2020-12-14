import yaml
import numpy as np
import cv2
import os.path
import time
import json
import requests
import time

class ImgProcessor(object):

        def __init__(self):
                self.car = 'storage\car.xml'
                
        def process_frame(self, fname):
                car = self.car
                #if os.path.exists(".\storage\config\id1.yml"):
                #if os.path.exists(".\\storage\\config\\" + fname):
                #parking_data = self.parking_datasets(".\storage\config\id1.yml")
                parking_data = self.parking_datasets(".\\storage\\config\\" + fname)
                #else:
                        #import lib.datasets
                        #parking_data = self.parking_datasets("storage\config\id1.yml")
                parking_contours, parking_bounding_rects, parking_mask, parking_data_motion = self.get_parking_info(parking_data)
                kernel_erode = self.set_erode_kernel()
                kernel_dilate = self.set_dilate_kernel()
                parking_status = [False]*len(parking_data)
                parking_buffer = [None]*len(parking_data)
                available_spots = self.main(car, parking_contours, parking_bounding_rects, parking_mask, parking_data_motion,
                                                                        kernel_erode, kernel_dilate, parking_status, parking_buffer, parking_data, fname)
                #print(available_spots)
                return available_spots[0], available_spots[1]
                
        def main(self, car, parking_contours, parking_bounding_rects, parking_mask, parking_data_motion,
                        kernel_erode, kernel_dilate, parking_status, parking_buffer, parking_data, fname):

                
                avail = []
                frame_pos = 0
                pos = 0.0
                fname = fname[:-4]
                
                

                car_cascade = cv2.CascadeClassifier(car)

                fgbg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
                timeout = time.time() + 5
                while True:
                        start = time.time()
                        r = requests.get('http://10.13.78.214:8080/get_frame', headers= {'cam_id': fname})
                        data = r.content
                        frame = json.loads(data.decode("utf8"))
                        frame = np.asarray(frame, np.uint8)
                        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                        first_frame = frame
                        frame_pos += 1
                        
                        # Smooth out the image, then convert to grayscale
                        blurImg = cv2.GaussianBlur(frame.copy(), (5,5), 3)
                        grayImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2GRAY)
                        line_img = frame.copy()
                        vpl = np.copy(line_img) * 0 #Virtual Parking Lot

                        # Drawing the Overlay. Text overlay at the left corner of screen
                        str_on_frame = "%d" % (frame_pos)
                        cv2.putText(line_img, str_on_frame, (5,30), cv2.FONT_HERSHEY_SIMPLEX,
                                                        0.8, (0,255,255), 2, cv2.LINE_AA)

                        


                        fgmask = fgbg.apply(blurImg)
                        bw = np.uint8(fgmask==255)*255
                        bw = cv2.erode(bw, kernel_erode, iterations=1)
                        bw = cv2.dilate(bw, kernel_dilate, iterations=1)

                        (cnts, _) = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # loop over the contours
                        for c in cnts:
                                if cv2.contourArea(c) < 500:
                                        continue
                                (x, y, w, h) = cv2.boundingRect(c)
                                cv2.rectangle(line_img, (x, y), (x + w, y + h), (255, 0, 0), 1)

                #Use the classifier to detect cars and help determine which parking spaces are available and unavailable
                        avail = self.detection(parking_bounding_rects, parking_data, parking_status,
                                          parking_buffer, grayImg, start, parking_mask, line_img, car_cascade, vpl)


                        

                        # Display video
                        #cv2.imshow('frame', line_img)
                        
                        #k = cv2.waitKey(1)
                        if time.time() > timeout:
                        #if k == ord('q'):
                                break

                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                return avail[0], avail[1]
        
        def run_classifier(self, img, id, car_cascade):
                cars = car_cascade.detectMultiScale(img, 1.1, 1)
                if cars == ():
                        return False
                else:
                        return True


        def parking_datasets(self, fn_yaml):
                with open(fn_yaml, 'r') as stream:
                        parking_data = yaml.load(stream)
                return parking_data

        def yaml_loader(self, file_path):
                with open(file_path, "r") as file_descr:
                        data = yaml.load(file_descr)
                        return data

        def yaml_dump(self,file_path, data):
                with open(file_path, "a") as file_descr:
                        yaml.dump(data, file_descr)
                        
        def get_parking_info(self, parking_data):
                parking_contours = []
                parking_bounding_rects = []
                parking_mask = []
                parking_data_motion = []
                if parking_data != None:
                        for park in parking_data:
                                points = np.array(park['points'])
                                rect = cv2.boundingRect(points)
                                points_shifted = points.copy()
                                points_shifted[:,0] = points[:,0] - rect[0] # shift contour to region of interest
                                points_shifted[:,1] = points[:,1] - rect[1]
                                parking_contours.append(points)
                                parking_bounding_rects.append(rect)
                                mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1,
                                                                                        color=255, thickness=-1, lineType=cv2.LINE_8)
                                mask = mask==255
                                parking_mask.append(mask)
                return parking_contours, parking_bounding_rects, parking_mask, parking_data_motion;

        def set_erode_kernel(self):
                kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
                return kernel_erode

        def set_dilate_kernel(self):
                kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,(5,19))
                return kernel_dilate


        def print_parkIDs(self, park, points, line_img, car_cascade, parking_bounding_rects, grayImg, parking_data, parking_status, vpl):

                spots_change = 0
                total_spots = len(parking_data)
                avail = []
                for ind, park in enumerate(parking_data):
                        points = np.array(park['points'])
                        if parking_status[ind]:
                                color = (0,255,0)
                                spots_change += 1
                                spot = 'Available'
                                rect = parking_bounding_rects[ind]
                                roi_gray_ov = grayImg[rect[1]:(rect[1] + rect[3]),
                                                           rect[0]:(rect[0] + rect[2])]  # crop roi for faster calcluation
                                cars = car_cascade.detectMultiScale(roi_gray_ov, 1.1, 1)
                                if cars == ():
                                        res = False
                                else:
                                        res = True
                                if res:
                                        parking_data_motion.append(parking_data[ind])
                                        
                                        color = (0,0,255)
                        else:
                                color = (0,0,255)
                                spot = 'Unavailable'
                        avail.append(spot)
                        
                        cv2.drawContours(line_img, [points], contourIdx=-1,
                                                                 color=color, thickness=2, lineType=cv2.LINE_8)
                        cv2.drawContours(vpl, [points], contourIdx=-1,
                                                                 color=color, thickness=2, lineType=cv2.LINE_8)
                        
                        moments = cv2.moments(points)
                        centroid = (int(moments['m10']/moments['m00'])-3, int(moments['m01']/moments['m00'])+3)
                        # putting numbers on marked regions
                        cv2.putText(line_img, str(park['id']), (centroid[0]+1, centroid[1]+1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                        cv2.putText(line_img, str(park['id']), (centroid[0]-1, centroid[1]-1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                        cv2.putText(line_img, str(park['id']), (centroid[0]+1, centroid[1]-1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                        cv2.putText(line_img, str(park['id']), (centroid[0]-1, centroid[1]+1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                        cv2.putText(line_img, str(park['id']), centroid, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

                        cv2.putText(vpl, str(park['id']), (centroid[0]+1, centroid[1]+1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                        cv2.putText(vpl, str(park['id']), (centroid[0]-1, centroid[1]-1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                        cv2.putText(vpl, str(park['id']), (centroid[0]+1, centroid[1]-1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                        cv2.putText(vpl, str(park['id']), (centroid[0]-1, centroid[1]+1), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
                        cv2.putText(vpl, str(park['id']), centroid, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

                
                        
                

                #Display number of available parking spaces on video for each frame change.
                spots_on_frame = "%d/%d" % (spots_change, total_spots)
                cv2.putText(line_img, spots_on_frame  + ' spaces are available', (6,61), cv2.FONT_HERSHEY_COMPLEX,
                                                        0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(line_img, spots_on_frame  + ' spaces are available', (4,59), cv2.FONT_HERSHEY_COMPLEX,
                                                        0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(line_img, spots_on_frame  + ' spaces are available', (6,59), cv2.FONT_HERSHEY_COMPLEX,
                                                        0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(line_img, spots_on_frame  + ' spaces are available', (4,61), cv2.FONT_HERSHEY_COMPLEX,
                                                        0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(line_img, spots_on_frame  + ' spaces are available', (5,60), cv2.FONT_HERSHEY_COMPLEX,
                                                        0.7, (0,0,0), 2, cv2.LINE_AA)

                return avail


        def detection(self, parking_bounding_rects, parking_data, parking_status, parking_buffer, grayImg, start, parking_mask, line_img, car_cascade, vpl):

                avail = []

                # detecting cars and vacant spaces
                #print(parking_data['points'])
                for ind, park in enumerate(parking_data):
                        points = np.array(park['points'])
                        #print(points)
                        rect = parking_bounding_rects[ind]

                        roi_gray = grayImg[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])] # crop roi for faster calcluation

                        laplacian = cv2.Laplacian(roi_gray, cv2.CV_64F)
                        
                        points[:,0] = points[:,0] - rect[0] # shift contour to roi
                        points[:,1] = points[:,1] - rect[1]
                        delta = np.mean(np.abs(laplacian * parking_mask[ind]))
                        
                        pos = time.time()
                        
                        status = delta < 2.2
                        # If detected a change in parking status, save the current time
                        if status != parking_status[ind] and parking_buffer[ind]==None:
                                parking_buffer[ind] = pos
                                
                        # If status is still different than the one saved and counter is open
                        elif status != parking_status[ind] and parking_buffer[ind]!=None:
                                if pos - parking_buffer[ind] > 1:
                                        parking_status[ind] = status
                                        parking_buffer[ind] = None
                        # If status is still same and counter is open
                        elif status == parking_status[ind] and parking_buffer[ind]!=None:
                                parking_buffer[ind] = None

        # changing the color on the basis on status change occured in the above section and putting numbers on areas
                        lot_id = park['lot']
                        #print(lot_id)
                        avail = self.print_parkIDs(park, points, line_img, car_cascade,
                                                  parking_bounding_rects, grayImg, parking_data, parking_status, vpl)
                #print(type(lot_id))
                return avail, lot_id
          
        
