from keras.models import load_model
from keras.datasets import cifar100
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pickle

## LOAD DATA ##
print("Loading models...")
baseline = load_model('./new_dataset/models/keras.keras')
# alexnet = load_model('./new_dataset/models/alexnet.keras')
# lenet = load_model('./new_dataset/models/lenet.keras')
vgg16 = load_model('./new_dataset/models/keras-VGG16-cifar10.keras')

print('Loading history...')
with open('./new_dataset/history/keras', "rb") as file_pi:
    baselineHistory = pickle.load(file_pi)
# with open('./new_dataset/history/alexnet', "rb") as file_pi:
#     alexnetHistory = pickle.load(file_pi)
# with open('./new_dataset/history/lenet', "rb") as file_pi:
#     lenetHistory = pickle.load(file_pi)
with open('./new_dataset/history/vgg16', "rb") as file_pi:
    vgg16History = pickle.load(file_pi)

print('Loading data...')
(X_train, y_train), (X_test, y_test) = cifar100.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
reshaped_y_test = y_test.reshape(-1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def confusion_matrix(model, y_test):
    y_predictions= model.predict(X_test)
    # y_test.reshape(-1,)
    # y_predictions.reshape(-1,)
    y_predictions= np.argmax(y_predictions, axis=1)
    y_test= np.argmax(y_test, axis=1)

    from sklearn.metrics import confusion_matrix, accuracy_score
    plt.figure(figsize=(7, 6))
    plt.title('Confusion matrix', fontsize=16)
    plt.imshow(confusion_matrix(y_test, y_predictions))
    # plt.xticks(np.arange(10), classes, rotation=45, fontsize=12)
    # plt.yticks(np.arange(10), classes, fontsize=12)
    plt.colorbar()
    plt.show()

confusion_matrix(baseline, y_test)
# confusion_matrix(alexnet)
# confusion_matrix(lenet)
confusion_matrix(vgg16, y_test)



def predictions(model, y_test):
    y_predictions= model.predict(X_test)
    # y_test.reshape(-1,)
    # y_predictions.reshape(-1,)
    y_predictions= np.argmax(y_predictions, axis=1)
    y_test= np.argmax(y_test, axis=1)

    L = 8
    W = 8
    fig, axes = plt.subplots(L, W, figsize = (20,20))
    axes = axes.ravel() # 

    for i in np.arange(0, L * W):  
        axes[i].imshow(X_test[i])
        axes[i].set_title("Predicted = {}\n Actual  = {}".format(y_predictions[i], y_test[i]))
        axes[i].axis('off')

    plt.subplots_adjust(wspace=1)
    plt.show()

predictions(baseline, y_test)
# predictions(alexnet)
# predictions(lenet)
predictions(vgg16, y_test)
