########################## importing all the modules ###########################
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

import cv2
import pyserial

model = 'keras_model.h5'
image = 'test_photo.jpg'

############################ Declearing the pyserial port ######################
ser = serial.Serial('COM5', 9600)

############################ Keras Prediction ######################
def kerasPredict(model, image):
  np.set_printoptions(suppress=True)
  model = tensorflow.keras.models.load_model()
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  image = Image.open()
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.ANTIALIAS)
  image_array = np.asarray(image)
  image.show()
  normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
  data[0] = normalized_image_array
  prediction = model.predict(data)
  print(prediction)
  return prediction

############################# main function ####################################
cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    cv2.imshow('frame', img)
    preduction = kerasPredict(model,img)

    if(preduction == forward):
        time.sleep(0.1)
        ser.write("0") # forward
    elif(preduction == right):
        time.sleep(0.1)
        ser.write("-1") # right
    elif(preduction == left):
        time.sleep(0.1)
        ser.write("1") # left
    elif(preduction == stop):
        time.sleep(0.1)
        ser.write("10") # stop

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
