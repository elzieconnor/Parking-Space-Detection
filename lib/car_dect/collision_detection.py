import yaml
import numpy as np



with open("car_coor.yml", 'r') as car_yaml:
    car_coor = yaml.safe_load(car_yaml)

with open("parking_spots.yml", 'r') as parking_yaml:
    spot_Coor = yaml.safe_load(parking_yaml)

def intersect_over_union(car_coor, spot_Coor):
    #for index in range(len(car_coor)):
    #    for key in car_coor[index]:
    #        print(car_coor[index][key])
    #Assigning each car coordinate to a variable
    carCoor = car_coor[0]['coors']
    carCoor_x1 = car_coor[0]['coors'][0][0]
    carCoor_y1 = car_coor[0]['coors'][0][1]
    carCoor_x2 = car_coor[0]['coors'][1][0]
    carCoor_y2 = car_coor[0]['coors'][1][1]
    #print(carCoor_x1)


    #for i in range(len(spot_Coor)):
    #    for k in spot_Coor[i]:
            #print(spot_Coor[i][k])
    #Assigning each parking spot coordinate to variable
    spotCoor = [spot_Coor[0]['points'][2], spot_Coor[0]['points'][0]]
    spotCoor_x1 = spot_Coor[0]['points'][2][0]
    spotCoor_y1 = spot_Coor[0]['points'][2][1]
    spotCoor_x2 = spot_Coor[0]['points'][0][0]
    spotCoor_y2 = spot_Coor[0]['points'][0][1]
    #print(spotCoor_x2)

    
    xA = max(carCoor_x1, spotCoor_x1)
    yA = max(carCoor_y1, spotCoor_y1)
    xB = min(carCoor_x2, spotCoor_x2)
    yB = min(carCoor_y2, spotCoor_y2)

    interArea = (xB - xA + 1) * (yB - yA + 1)
    carBoxArea = (carCoor_x2 - carCoor_x1 + 1) * (carCoor_y2 - carCoor_y1 + 1)
    spotBoxArea = (spotCoor_x2 - spotCoor_x1 + 1) * (spotCoor_y2 - spotCoor_y1 + 1)

    iou = interArea / float(carBoxArea + spotBoxArea - interArea)

    return iou
    

if __name__ == '__main__':
    print(intersect_over_union(car_coor, spot_Coor))
