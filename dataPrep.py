import cv2
import os
import random
import pickle
import numpy as np


folder_path = "pre faces"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

min_faceSize = 224
last_face_size = 224

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

training_data = []

catagori_index =0

for filename1 in os.listdir(folder_path):
    i = 0
    for filename2 in os.listdir(os.path.join(folder_path,filename1)):
        img = cv2.imread(os.path.join(os.path.join(folder_path,filename1), filename2), cv2.COLOR_BGR2RGB)
        if img is not None:

            if (img is None):
                print("resim acılamadı")
                exit(0)

            faces = face_cascade.detectMultiScale(img, 1.3, 3, minSize=(min_faceSize, min_faceSize))
            if (faces is None):
                print('yüz bulunamadı')
                exit(0)

            for (x, y, w, h) in faces:

                faceimg = img[y:y + h, x:x + w]
                lastimg = cv2.resize(faceimg, (last_face_size, last_face_size))
                tmp = "post faces/"+filename1
                if not os.path.isdir(tmp):
                    os.mkdir(tmp)
                cv2.imwrite("post faces/"+filename1+"/"+str(i)+".jpg", lastimg)
                i=i+1

                try:

                    training_data.append([lastimg, catagori_index])
                except Exception as e:
                    pass

    catagori_index+=1

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, last_face_size, last_face_size, 3)

pickle_out = open("x.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

