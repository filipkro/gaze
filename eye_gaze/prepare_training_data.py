import csv
import cv2
import numpy as np
from augmenter import augmenter
import random

def show_images(data):
    if('croppedLeft' in data):
        croppedLeft = data['croppedLeft']
        cl_center = data['cl_center']
        cl_inner_corner = data['cl_inner_corner']
        cl_outer_corner = data['cl_outer_corner']
        cv2.circle(croppedLeft, (int(np.round(cl_center[0])),int(np.round(cl_center[1]))), 1, (0,0,255))
        cv2.circle(croppedLeft, (int(np.round(cl_outer_corner[0])),int(np.round(cl_outer_corner[1]))), 0, (0,255,255))
        cv2.circle(croppedLeft, (int(np.round(cl_inner_corner[0])),int(np.round(cl_inner_corner[1]))), 0, (0,255,255))

        scale_percent = 100 # percent of original size
        width = int(croppedLeft.shape[1] * scale_percent / 100)
        height = int(croppedLeft.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(croppedLeft, dim, interpolation = cv2.INTER_AREA)
        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE) 
        cv2.imshow('image',resized)
        cv2.waitKey(0)

    if('croppedRight' in data):
        croppedRight = data['croppedRight']
        cr_center = data['cr_center']
        cr_inner_corner = data['cr_inner_corner']
        cr_outer_corner = data['cr_outer_corner']
        cv2.circle(croppedRight, (int(np.round(cr_center[0])),int(np.round(cr_center[1]))), 1, (0,0,255))
        cv2.circle(croppedRight, (int(np.round(cr_outer_corner[0])),int(np.round(cr_outer_corner[1]))), 0, (0,255,255))
        cv2.circle(croppedRight, (int(np.round(cr_inner_corner[0])),int(np.round(cr_inner_corner[1]))), 0, (0,255,255))


        scale_percent = 100 # percent of original size
        width = int(croppedRight.shape[1] * scale_percent / 100)
        height = int(croppedRight.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(croppedRight, dim, interpolation = cv2.INTER_AREA)
        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE) 
        cv2.imshow('image',resized)
        cv2.waitKey(0)

def read_csv(file_path):
    reader = csv.DictReader(open(file_path))
    result = {}
    for row in reader:
        for column, value in row.items():  # consider .iteritems() for Python 2
            if column != "Image":
                if len(value)>0:
                    result.setdefault(column, []).append(float(value))
                else:
                    result.setdefault(column, []).append(0)
            else:
                result.setdefault(column, []).append(value)
    return result

def build_x_y(data, x, y):
    if('croppedLeft' in data):
        labels_left = [data['cl_center'][0],data['cl_center'][1],data['cl_inner_corner'][0],data['cl_inner_corner'][1],
            data['cl_outer_corner'][0],data['cl_outer_corner'][1]]
        x.append(np.reshape(data['croppedLeft'],(32,32,1)))
        y.append(np.array(labels_left))

    if('croppedRight' in data):
        labels_right = [data['cr_center'][0],data['cr_center'][1],data['cr_inner_corner'][0],data['cr_inner_corner'][1],
            data['cr_outer_corner'][0],data['cr_outer_corner'][1]]
        x.append(np.reshape(data['croppedRight'],(32,32,1)))
        y.append(np.array(labels_right))




if __name__ == "__main__":

    result = read_csv('Data/training/training.csv')
    images = result['Image']
    left_eye_center = list(zip(result['left_eye_center_x'],result["left_eye_center_y"]))
    right_eye_center = list(zip(result["right_eye_center_x"],result["right_eye_center_y"]))
    left_eye_inner_corner = list(zip(result['left_eye_inner_corner_x'],result["left_eye_inner_corner_y"]))
    left_eye_outer_corner = list(zip(result['left_eye_outer_corner_x'],result["left_eye_outer_corner_y"]))
    right_eye_inner_corner = list(zip(result['right_eye_inner_corner_x'],result["right_eye_inner_corner_y"]))
    right_eye_outer_corner = list(zip(result['right_eye_outer_corner_x'],result["right_eye_outer_corner_y"]))
    aug = augmenter(images,left_eye_center,left_eye_inner_corner,left_eye_outer_corner,right_eye_center,right_eye_inner_corner,right_eye_outer_corner)


    print("Generating training data")
    size = len(images)
    x = []
    y = []

    for i in range(size):
        if(i%100==0):
            print(i,"/",size)
        data = aug.process_image(i,0,0)
        build_x_y(data,x,y)

        for k in range(4):
            r1 = random.randint(-5,5)
            r2 = random.randint(-5,5)
            data = aug.process_image(i,r1,r2)
            build_x_y(data,x,y)
    print("Data generated")
    x = np.array(x)
    y = np.array(y)
    print("X:",np.shape(x))
    print("Y:",np.shape(y))
    np.savez("Data/generated_training", x=x, y=y)
