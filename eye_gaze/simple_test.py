import prepare_training_data
from tensorflow import keras
from coordconv.coord import CoordinateChannel2D
import csv
import cv2
import numpy as np
from train_neural_net import load

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

def draw_landmarks(image, landmarks, ground_truth = False):
    if(not ground_truth):
        color1 = (0,0,255)
    else:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        color1 = (0,255,255)
    image = cv2.circle(image, (int(np.round(landmarks[0])),int(np.round(landmarks[1]))), 0, color1)
    image = cv2.circle(image, (int(np.round(landmarks[2])),int(np.round(landmarks[3]))), 0, color1)
    image = cv2.circle(image, (int(np.round(landmarks[4])),int(np.round(landmarks[5]))), 0, color1)
    return image

def convert2image(pixels):
    img = np.array(pixels.split())
    img = img.reshape(96,96)  # dimensions of the image
    return img.astype(np.uint8) # return the image

if __name__ == "__main__":
    model = keras.models.load_model('Data/Models/CorCNN.model')
    x,y = load("Data/generated/generated_test.npz")
    for i in range(30):
        image = x[i]
        input = np.expand_dims(x[i], axis=0)
        pred = model.predict(input)
        print(pred)
        image = draw_landmarks(image,y[i],ground_truth=True)
        cv2.imshow('image',draw_landmarks(image,pred[0]))
        cv2.waitKey(0)

"""
    model = keras.models.load_model('Data/Models/CorCNN.model')
    result = read_csv('Data/test/test.csv')
    for i in range(10):
        pixels = result['Image'][i]
        image = convert2image(pixels)
        x = 15
        y = 20
        eye = image[y:y+32,x:x+32]
        input = np.reshape(eye,(32,32,1))
        input = np.expand_dims(input, axis=0)
        pred = model.predict(input)
        print(pred)
        cv2.imshow('image',draw_landmarks(eye,pred[0]))
        cv2.waitKey(0)
"""
