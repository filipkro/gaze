import numpy as np
import cv2

class augmenter:

    def __init__(self,images,left_eye_center,left_eye_inner_corner,left_eye_outer_corner,right_eye_center,
    right_eye_inner_corner,right_eye_outer_corner):

        self.images = images

        self.left_eye_center = left_eye_center
        self.left_eye_inner_corner = left_eye_inner_corner
        self.left_eye_outer_corner = left_eye_outer_corner

        self.right_eye_center = right_eye_center
        self.right_eye_inner_corner = right_eye_inner_corner
        self.right_eye_outer_corner = right_eye_outer_corner


    def convert2image(self,pixels):
        img = np.array(pixels.split())
        img = img.reshape(96,96)  # dimensions of the image
        return img.astype(np.uint8) # return the image

    def process_image(self, i, off_x, off_y):
        lec = self.left_eye_center[i][0] + self.left_eye_center[i][1]
        lec1 = self.left_eye_outer_corner[i][0] + self.left_eye_outer_corner[i][1]
        lec2 = self.left_eye_inner_corner[i][0] + self.left_eye_inner_corner[i][0]

        rec = self.right_eye_center[i][0] + self.right_eye_center[i][1]
        rec1 = self.right_eye_outer_corner[i][0] + self.right_eye_outer_corner[i][1]
        rec2 = self.right_eye_inner_corner[i][0] + self.right_eye_inner_corner[i][0]

        left_true =  (lec > 0 and lec1 > 0 and lec2 > 0)
        right_true = (rec > 0 and rec1 > 0 and rec2 > 0)

        processed_data = {}

        #Left eye
        if left_true:
            centerLeft = ((self.left_eye_outer_corner[i][0]+self.left_eye_inner_corner[i][0])/2, (self.left_eye_outer_corner[i][1]+self.left_eye_inner_corner[i][1])/2)
            cornerLeft = (max(int(np.round(centerLeft[0]-16+off_x)),0), max(int(np.round(centerLeft[1])-16+off_y),0))

            image = self.convert2image(self.images[i])
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            croppedLeft= image[cornerLeft[1]:cornerLeft[1]+32,cornerLeft[0]:cornerLeft[0]+32]

            cl_center = (self.left_eye_center[i][0]-cornerLeft[0],self.left_eye_center[i][1]-cornerLeft[1])
            cl_inner_corner = (self.left_eye_inner_corner[i][0]-cornerLeft[0],self.left_eye_inner_corner[i][1]-cornerLeft[1])
            cl_outer_corner = (self.left_eye_outer_corner[i][0]-cornerLeft[0],self.left_eye_outer_corner[i][1]-cornerLeft[1])

            processed_data['croppedLeft'] = croppedLeft
            processed_data['cl_center'] = cl_center
            processed_data['cl_inner_corner'] = cl_inner_corner
            processed_data['cl_outer_corner'] = cl_outer_corner

        #Right eye
        if right_true:
            centerRight = ((self.right_eye_outer_corner[i][0]+self.right_eye_inner_corner[i][0])/2, (self.right_eye_outer_corner[i][1]+self.right_eye_inner_corner[i][1])/2)
            cornerRight = (max(int(np.round(centerRight[0]-16+off_x)),0), max(int(np.round(centerRight[1])-16+off_y),0))

            image = self.convert2image(self.images[i])
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            croppedRight= image[cornerRight[1]:cornerRight[1]+32,cornerRight[0]:cornerRight[0]+32]

            cr_center = (self.right_eye_center[i][0]-cornerRight[0],self.right_eye_center[i][1]-cornerRight[1])
            cr_inner_corner = (self.right_eye_inner_corner[i][0]-cornerRight[0],self.right_eye_inner_corner[i][1]-cornerRight[1])
            cr_outer_corner = (self.right_eye_outer_corner[i][0]-cornerRight[0],self.right_eye_outer_corner[i][1]-cornerRight[1])

            processed_data['croppedRight'] = croppedRight
            processed_data['cr_center'] = cr_center
            processed_data['cr_inner_corner'] = cr_inner_corner
            processed_data['cr_outer_corner'] = cr_outer_corner

        return processed_data

