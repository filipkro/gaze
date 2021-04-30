# import libraries 
from mtcnn.mtcnn import MTCNN
import cv2
import os
import numpy as np
from eye_gaze import eye_gaze

def draw(image, face):
    bbox = face['box']
    cv2.putText(image,str(np.round(face['confidence'],2)),(int(np.round(bbox[0])),int(np.round(bbox[1]))),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
    cv2.rectangle(image, (int(np.round(bbox[0])),int(np.round(bbox[1]))),(int(np.round(bbox[2]+bbox[0])),int(np.round(bbox[3]+bbox[1]))),(0,0,255))

    for value in face['keypoints'].items():
        cv2.circle(image, (int(np.round(value[1][0])),int(np.round(value[1][1]))), 3, (0,0,255))
    return image

def draw_normal(image, normal, eye):
    p1 = (eye[0], eye[1])
    p2 = (int(p1[0]+normal[0]), int(p1[1]+normal[1]))

    cv2.line(image, p1, p2, (0,0,255), 2)
    return image

def crop_image(image, eye_center, size):
    corner = (eye_center[0]-16,eye_center[1]-size/2)
    cropped_image = image[int(np.round(corner[1])):int(np.round(corner[1]+size)),int(np.round(corner[0])):int(np.round(corner[0]+size))]
    
    return cropped_image


if __name__ == '__main__':

    # create the detector, using default weights
    detector = MTCNN()

    # define a video capture object 
    vid = cv2.VideoCapture(1) 
    counter = 0
    
    while(True): 
        if counter % 5 == 0:
            # Capture the video frame 
            # by frame 
            ret, frame = vid.read() 
            
            frame = cv2.flip(frame, 1)

            # detect faces in the image
            faces = detector.detect_faces(frame)
            
            while len(faces) == 0:
                ret, frame = vid.read()
                faces = detector.detect_faces(frame)
                frame = cv2.flip(frame, 1)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            for face in faces:
                keypoints = face['keypoints']
                left_eye = keypoints['left_eye']
                right_eye = keypoints['right_eye']

                left_cropped = crop_image(frame,left_eye,32)
                left_cropped = cv2.cvtColor(left_cropped,cv2.COLOR_RGB2GRAY)
                input_data_left = np.reshape(left_cropped,(32,32,1))

                right_cropped = crop_image(frame,right_eye,32)
                right_cropped = cv2.cvtColor(right_cropped,cv2.COLOR_RGB2GRAY)
                input_data_right = np.reshape(right_cropped,(32,32,1))

                path = 'D:/Python_Workspace/gaze_tracker/eye_gaze/Data/Models/CorCNN_98.model'

                eg = eye_gaze(path)

                left_dps = eg.calc_dps(input_data_left)[0]
                left_dps[0] = int(round(left_dps[0]))
                left_dps[1] = int(round(left_dps[1]))

                right_dps = eg.calc_dps(input_data_right)[0]
                right_dps[0] = int(round(right_dps[0]))
                right_dps[1] = int(round(right_dps[1]))

                frame = draw_normal(frame, 50*left_dps, left_eye)
                frame = draw_normal(frame, 50*right_dps, right_eye)
                # Display the resulting frame 
                cv2.imshow('frame', frame) 

            # the 'q' button is set as the 
            # quitting button you may use any 
            # desired button of your choice 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

    # After the loop release the cap object 
    vid.release() 
    # Destroy all the windows 
    cv2.destroyAllWindows()