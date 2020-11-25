import tensorflow as tf
import keras
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout,SeparableConv2D
from keras.optimizers import Adam
from keras.activations import softmax
from keras.utils import to_categorical
import pickle
from keras.callbacks import TensorBoard
import numpy as np
import time
from sklearn.utils import class_weight


name = "Test-"+str(time.time())

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))



class_weight = class_weight.compute_class_weight("balanced",np.unique(y),y)

y = to_categorical(y)

vgg19_model = keras.applications.vgg19.VGG19()

model = Sequential()

for layer in vgg19_model.layers:

    model.add(layer)

model.layers.pop()

for layer in model.layers:
    layer.trainable = False

model.add(
    Dense(4,activation="softmax")
)

model.compile(optimizer=Adam(lr=0.1),loss="categorical_crossentropy",metrics=["accuracy"])
tensorboard = TensorBoard(log_dir="logs/{}".format(name))
model.fit(X,y,batch_size=20,epochs=18,verbose=2,callbacks=[tensorboard],validation_split=0.3,class_weight=class_weight)
model.save_weights("TrainedModel.model")