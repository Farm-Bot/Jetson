########################## importing all the modules ###########################
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2,time
import serial


####################### setting up the tenserflow ##############################
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

numberOfDivision = 5
height, width, channels = img.shape
indi_height = height//numberOfDivision
indi_width = width//numberOfDivision

######################## classify the individual sub-image #####################
def predict(loc):
    image = Image.open(loc)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    pred = np.argmax(prediction, axis = 1)[:5]
    return(pred)

########################### splits the image into sub-images ###################
def split_predect(img):
    results =[]
    current_height = 0

    for xAxis in range(numberOfDivision):
        current_width = 0
        tem_array = []

        for yAxis in range(numberOfDivision):
            crop_img = img[current_height:current_height+indi_height, current_width:current_width+indi_width]
            cv2.imwrite('temp.jpg', crop_img)
            pre = [predict('temp.jpg')]
            tem_array.append(pre)
            current_width+=indi_width
        current_height+=indi_height
        results.append(tem_array)

    for i in results:
      print(i)
    return(results)

############################ Declearing the pyserial port ######################
ser = serial.Serial('COM5', 9600)

############################ declearing the array ##############################

forward = [[1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,0,1,1],
           [0,0,0,0,0],
           [0,0,0,0,0]
          ]

right =   [[1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,0,1],
           [0,0,0,0,0],
           [0,0,0,0,0]
          ]

left =    [[1,1,1,1,1],
           [1,1,1,1,1],
           [1,0,1,1,1],
           [0,0,0,0,0],
           [0,0,0,0,0]
          ]

stop =    [[1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1],
           [1,1,1,1,1],
           [0,0,0,0,0]
          ]
############################# main function ####################################


cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()
    cv2.imshow('frame', img)
    preduction = split_predect(img)

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
