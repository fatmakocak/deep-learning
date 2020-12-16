import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils


(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.interactive(False)
#datasette ki bir kaç örneği inceleyelim
fig = plt.figure()
for i in range(4):
  plt.subplot(3, 3, i+1) #x=satır sayısı , y=sütun sayısı ,z=index numarası
  plt.tight_layout() # grafikteki eksenleri otomatik olarak ayarlamak
  plt.imshow(X_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i])) #eksenler için başlık ayarı
  plt.xticks([])
  plt.yticks([])
fig.show()
#sinir ağımızı görüntüleri sınıflandıracak şekilde eğitmek için öncelikle genişlik ve yükseklik
#pixellerini büyük bir vektöre açmamız gerek .bu yüzden uzunluk 28x28 =784 olmalı.
#pixel değerimizin dağılımını grafikte çizelim
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(X_train[0], cmap='gray', interpolation='none')
plt.title("Digit: {}".format(y_train[0]))
plt.xticks([])
plt.yticks([])
plt.subplot(2, 1, 2)
plt.hist(X_train[0].reshape(784))
plt.title("Pixel Value Distribution")
fig.show()
#yeniden şekillendirmeden ve normalleştirmeden önce şekli basalım
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# 28x28 pixelden girdi vektörü oluşturalım
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# eğitime yardımcı olması için verileri normalleştirme(0-1 arası bir değer)
X_train /= 255
X_test /= 255

# eğitime hazır son giriş şeklini yazdır
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)

print(np.unique(y_train, return_counts=True))

n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)
#sıralı model ile doğrusal bir katman yığını oluşturma
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))
#sıralı modeli derleme
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
history = model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=2, validation_data=(X_test, Y_test))

# modeli kaydetme
save_dir = 'resultss/'
model_name = 'keras_mnist.h5'
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# çizme
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()

fig.show()
#modeli yükle
mnist_model = load_model("resultss/keras_mnist.h5")
loss_and_metrics = mnist_model.evaluate(X_test, Y_test, verbose=2)

print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])

# modeli yükleme ve tahminler oluşturma
mnist_model = load_model("resultss/keras_mnist.h5")
predicted_classes = mnist_model.predict_classes(X_test)

# hangisini doğru tahmin ettiğimize bakalım
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
print()
print(len(correct_indices), " classified correctly")
print(len(incorrect_indices), " classified incorrectly")

# şekil boyutunu ayarlama
plt.rcParams['figure.figsize'] = (7, 14)

figure_evaluation = plt.figure()

# 9 doğru tahmin
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6, 3, i+1)
    plt.imshow(X_test[correct].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted: {}, Truth: {}".format(predicted_classes[correct],
                                        y_test[correct]));\
    plt.xticks([])
    plt.yticks([])

# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6, 3, i+10)
    plt.imshow(X_test[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted {}, Truth: {}".format(predicted_classes[incorrect],
                                       y_test[incorrect]))
    plt.xticks([])
    plt.yticks([])

figure_evaluation.show()



