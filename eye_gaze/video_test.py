import cv2
import numpy as np
from eye_gaze import eye_gaze

def crop_image(image, corner, size):
    cropped_image = image[int(np.round(corner[1])):int(np.round(corner[1]+size)),int(np.round(corner[0])):int(np.round(corner[0]+size))]
    resized = cv2.resize(cropped_image, (32,32), interpolation = cv2.INTER_AREA)
    return resized

def draw_normal(image, normal, eye):
    normal *= 5
    p1 = (eye[0], eye[1])
    p2 = (int(p1[0]+normal[0]), int(p1[1]+normal[1]))

    cv2.line(image, p1, p2, (0,0,255), 2)
    return image

def draw_landmarks(image, points, corner):
    image = cv2.circle(image, (int(np.round(points[0]*3+corner[0])),int(np.round(points[1]*3+corner[1]))), 1, (0,255,255))
    image = cv2.circle(image, (int(np.round(points[2]*3+corner[0])),int(np.round(points[3]*3+corner[1]))), 1, (0,255,255))
    image = cv2.circle(image, (int(np.round(points[4]*3+corner[0])),int(np.round(points[5]*3+corner[1]))), 1, (0,255,255))

if __name__ == '__main__':

    cap = cv2.VideoCapture('Data/vid.mp4')
    ret, pixels = cap.read()
    i = 0
    gaze = eye_gaze('Data/Models/CorCNN.model')
    while True:
        #if(i%2==0):
        ret, pixels = cap.read()
        #i+=1
        image = cv2.resize(pixels, (360,640), interpolation = cv2.INTER_AREA)
        #image = cv2.flip(image, 1)
        lc = (65,215)
        rc = (175,215)
        left_eye = crop_image(image, lc, 96)
        left_eye = cv2.cvtColor(left_eye, cv2.COLOR_RGB2GRAY)
        left_eye = np.reshape(left_eye,(32,32,1))
        right_eye = crop_image(image, rc, 96)
        right_eye = cv2.cvtColor(right_eye, cv2.COLOR_RGB2GRAY)
        right_eye = np.reshape(right_eye,(32,32,1))
        dps_left, pred_left = gaze.calc_dps(left_eye)
        left_pupil = (lc[0]+int(np.round(pred_left[0]*3)),lc[1]+int(np.round(pred_left[1]*3)))
        draw_normal(image,dps_left,left_pupil)
        draw_landmarks(image,pred_left,lc)
        dps_right, pred_right = gaze.calc_dps(right_eye)
        right_pupil = (rc[0]+int(np.round(pred_right[0]*3)),rc[1]+int(np.round(pred_right[1]*3)))
        draw_normal(image,dps_right,right_pupil)
        draw_landmarks(image,pred_right,rc)
        cv2.imshow('pixels',image)
        if cv2.waitKey(0) == 27:
            break  # esc to quit
        
    cap.release()
    cv2.destroyAllWindows()
