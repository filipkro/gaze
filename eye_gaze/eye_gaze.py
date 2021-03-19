import csv
import cv2
import numpy as np

def convert2image(pixels):
    img = np.array(pixels.split())
    img = img.reshape(96,96)  # dimensions of the image
    return img.astype(np.uint8) # return the image

reader = csv.DictReader(open('Data/training/training.csv'))

result = {}
for row in reader:
    for column, value in row.items():  # consider .iteritems() for Python 2
        result.setdefault(column, []).append(value)

images = result['Image']
left_eye_center_x = result['left_eye_center_x']
left_eye_center_y = result["left_eye_center_y"]
right_eye_center_x = result['right_eye_center_x']
right_eye_center_y = result["right_eye_center_y"]
for i in range(20):
    image = convert2image(images[i])
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    cv2.circle(image, (int(np.round(float(left_eye_center_x[i]))),int(np.round(float(left_eye_center_y[i])))), 3, (0,0,255))
    cv2.circle(image, (int(np.round(float(right_eye_center_x[i]))),int(np.round(float(right_eye_center_y[i])))), 3, (0,0,255))
    #print(images[0])
    cv2.imshow('image',image)
    cv2.waitKey(0)
cv2.destroyAllWindows()


def draw(image, face):

   bbox = face['box']
   cv2.putText(image,str(np.round(face['confidence'],2)),(int(np.round(bbox[0])),int(np.round(bbox[1]))),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
   cv2.rectangle(image, (int(np.round(bbox[0])),int(np.round(bbox[1]))),(int(np.round(bbox[2]+bbox[0])),int(np.round(bbox[3]+bbox[1]))),(0,0,255))

   for value in face['keypoints'].items():
      cv2.circle(image, (int(np.round(value[1][0])),int(np.round(value[1][1]))), 3, (0,0,255))
   return image