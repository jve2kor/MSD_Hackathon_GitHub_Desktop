import cv2
import sys
import numpy as np

from keras.models import load_model
model=load_model("/Users/jvr605/Downloads/FaceDetect_master/Train_keras.mod")

emotions = ['Angry','Disgust', 'Fear', 'Happy','Sad','Surprise','Neutral']
# Get user supplied values
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
                                     gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(30, 30),
                                     flags = cv2.cv.CV_HAAR_SCALE_IMAGE
                                     )

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for i,(x, y, w, h)  in enumerate(faces):
    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    sub_face = gray[y:y+h, x:x+w]
    r = 48.0 / sub_face.shape[1]
    dim = (48, int(sub_face.shape[0] * r))
    # perform the actual resizing of the image and show it
    resized = cv2.resize(sub_face, dim, interpolation = cv2.INTER_AREA)
    temp =resized
    #resized.reshape
    #resized = np.array([resized])
    resized =resized.reshape(-1,1,48,48)
    prediction_result = np.argmax(model.predict(resized))

    print emotions[prediction_result]

    print resized.shape
    cv2.imwrite("face-" + str(i)+".jpg",sub_face)



cv2.imshow("Faces found", temp)
cv2.waitKey(0)
