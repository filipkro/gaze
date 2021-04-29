from tensorflow import keras
import cv2
import numpy as np
from train_neural_net import load

class eye_gaze:

    def __init__(self,model_path):
        self.model = keras.models.load_model(model_path)

    def draw_normal(self,image, normal, pred):
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        p1 = (pred[0], pred[1])
        p2 = (int(np.round(p1[0]+normal[0])), int(np.round(p1[1]+normal[1])))
        print('p1',p1)
        print('p2',p2)
        print('normal',normal)
        cv2.line(image, p1, p2, (0,255,255), 1)
        return image

    def draw_landmarks(self, image, landmarks, ground_truth = False):
        if(not ground_truth):
            color1 = (0,0,255)
        else:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            color1 = (0,255,255)
        image = cv2.circle(image, (int(np.round(landmarks[0])),int(np.round(landmarks[1]))), 0, color1)
        image = cv2.circle(image, (int(np.round(landmarks[2])),int(np.round(landmarks[3]))), 0, color1)
        image = cv2.circle(image, (int(np.round(landmarks[4])),int(np.round(landmarks[5]))), 0, color1)
        return image

    def calc_dps(self,eye):
        image = eye.copy()
        eye = np.expand_dims(eye, axis=0)
        pred = self.model.predict(eye)[0]
        eye_mid_x = pred[2]+abs(pred[2]-pred[4])/2
        eye_mid_y = pred[3]+abs(pred[3]-pred[5])/2
        eye_mid = np.array([eye_mid_x,eye_mid_y])
        pupil = np.array([pred[0],pred[1]])
        dps = pupil-eye_mid
        dps = np.array([dps[0],dps[1],0])
        image = self.draw_normal(image,dps,pred)
        image = self.draw_landmarks(image,pred,ground_truth=False)
        cv2.imshow("test",image)
        cv2.waitKey(0)
        return dps