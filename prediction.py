import pickle
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import cv2
import itertools
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.utils.multiclass import unique_labels
from keras.models import load_model
import keras
from keras.applications.vgg19 import VGG19

X = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

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

model.load_weights("test.model")
CATEGORIES = ["adem","kaan","Muhammed","TunahanK"]

def prepare(filepath):
    IMG_SIZE = 224
    img_array = cv2.imread(filepath,cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return  new_array.reshape(-1,IMG_SIZE,IMG_SIZE,3)



predictions = model.predict_classes(X,batch_size=10,verbose=1)

print(predictions)

cm = confusion_matrix(y_true=y,y_pred=predictions)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


plot_confusion_matrix(cm,CATEGORIES,"Dogruluk Orani",plt.cm.Blues)




