from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pickle

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

## LOAD DATA ##
print("Loading models...")
baseline = load_model('./baselines/models/keras.keras')
alexnet = load_model('./baselines/models/alexnet.keras')
lenet = load_model('./baselines/models/lenet.keras')
vgg16 = load_model('./baselines/models/keras-VGG16-cifar10.keras')
# vgg19 = load_model('./baselines/models/vgg19.keras')

baselineImp = load_model('./improved/models/keras.keras')
alexnetImp = load_model('./improved/models/alexnet.keras')
lenetImp = load_model('./improved/models/lenet.keras')
vgg16Imp = load_model('./improved/models/keras-VGG16-cifar10.keras')
# vgg19Imp = load_model('./improved/models/vgg19.keras')

print('Loading history...')
with open('./baselines/history/keras', "rb") as file_pi:
    baselineHistory = pickle.load(file_pi)
with open('./baselines/history/alexnet', "rb") as file_pi:
    alexnetHistory = pickle.load(file_pi)
with open('./baselines/history/lenet', "rb") as file_pi:
    lenetHistory = pickle.load(file_pi)
with open('./baselines/history/vgg16', "rb") as file_pi:
    vgg16History = pickle.load(file_pi)
# with open('./baselines/history/vgg19', "rb") as file_pi:
#     vgg19History = pickle.load(file_pi)

with open('./improved/history/keras', "rb") as file_pi:
    baselineHistoryImp = pickle.load(file_pi)
with open('./improved/history/alexnet', "rb") as file_pi:
    alexnetHistoryImp = pickle.load(file_pi)
with open('./improved/history/lenet', "rb") as file_pi:
    lenetHistoryImp = pickle.load(file_pi)
with open('./improved/history/vgg16', "rb") as file_pi:
    vgg16HistoryImp = pickle.load(file_pi)
# with open('./improved/history/vgg19', "rb") as file_pi:
#     vgg19HistoryImp = pickle.load(file_pi)

print('Loading data...')
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
reshaped_y_test = y_test.reshape(-1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

barWidth = .1

br1 = np.arange(4)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

## VISUALIZATIONS ##

print('Creating bar chart one...')
plt.bar(br1, [
    baselineHistory["accuracy"][len(baselineHistory["accuracy"]) - 1] * 100,
    alexnetHistory["accuracy"][len(alexnetHistory["accuracy"]) - 1] * 100,
    lenetHistory["accuracy"][len(lenetHistory["accuracy"]) - 1] * 100,
    vgg16History["accuracy"][len(vgg16History["accuracy"]) - 1] * 100,
], width = barWidth, label='Baselines Training')
plt.bar(br2, [
        baseline.evaluate(X_test, y_test, verbose=1)[1] * 100,
        alexnet.evaluate(X_test, reshaped_y_test, verbose=1)[1] * 100,
        lenet.evaluate(X_test, reshaped_y_test, verbose=1)[1] * 100,
        vgg16.evaluate(X_test, y_test, verbose=1)[1] * 100,
        # vgg19.evaluate(X_test, y_test, verbose=1)[1] * 100,
    ], width = barWidth, label='Baselines Testing')
plt.bar(br3, [
    baselineHistoryImp["accuracy"][len(baselineHistoryImp["accuracy"]) - 1] * 100,
    alexnetHistoryImp["accuracy"][len(alexnetHistoryImp["accuracy"]) - 1] * 100,
    lenetHistoryImp["accuracy"][len(lenetHistoryImp["accuracy"]) - 1] * 100,
    vgg16HistoryImp["accuracy"][len(vgg16HistoryImp["accuracy"]) - 1] * 100,
], width = barWidth, label='Improved Training')
plt.bar(br4, [
        baselineImp.evaluate(X_test, y_test, verbose=1)[1] * 100,
        alexnetImp.evaluate(X_test, reshaped_y_test, verbose=1)[1] * 100,
        lenetImp.evaluate(X_test, reshaped_y_test, verbose=1)[1] * 100,
        vgg16Imp.evaluate(X_test, y_test, verbose=1)[1] * 100,
        # vgg19.evaluate(X_test, y_test, verbose=1)[1] * 100,
    ], width = barWidth, label='Improved Testing')
plt.xticks([r + barWidth for r in range(4)], ['keras', 'alexnet', 'lenet', 'vgg16'])
plt.legend()
plt.show()

## OUTPUT SCORES ##
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

# X_train = X_train / 255.0
# X_test = X_test / 255.0
# # one hot encode outputs
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# sparse_y_train = y_train.reshape(-1,)

# # Reshape converting 2D to 1D
# sparse_y_test = y_test.reshape(-1,)
# sparse_y_train = y_train.reshape(-1,)

# # This code normalazation
# sparse_x_train = X_train
# sparse_x_test = X_test

# improvedScores = improved.evaluate(X_test, y_test, verbose=0)
# print("Improved Accuracy: %.2f%%" % (improvedScores[1]*100))

# baselineScores = baseline.evaluate(X_test, y_test, verbose=0)
# print("Baseline Accuracy: %.2f%%" % (baselineScores[1]*100))

# # alexnetScores = alexnet.evaluate(sparse_x_test, sparse_y_test, verbose=1)
# # print("Alexnet Accuracy: %.2f%%" % (alexnetScores[1]*100))

# # lenetScores = lenet.evaluate(sparse_x_test, sparse_y_test, verbose=1)
# # print("Lenet Accuracy: %.2f%%" % (lenetScores[1]*100))

# vgg16Scores = vgg16.evaluate(X_test, y_test, verbose=0)
# print("VGG 16 Accuracy: %.2f%%" % (vgg16Scores[1]*100))

# ## SHOW PLOTS ##

# def confusion_matrix(model):
#     y_predictions= model.predict(X_test)
#     y_test.reshape(-1,)
#     y_predictions.reshape(-1,)
#     y_predictions= np.argmax(y_predictions, axis=1)
#     y_test= np.argmax(y_test, axis=1)

#     from sklearn.metrics import confusion_matrix, accuracy_score
#     plt.figure(figsize=(7, 6))
#     plt.title('Confusion matrix', fontsize=16)
#     plt.imshow(confusion_matrix(y_test, y_predictions))
#     plt.xticks(np.arange(10), classes, rotation=45, fontsize=12)
#     plt.yticks(np.arange(10), classes, fontsize=12)
#     plt.colorbar()
#     plt.show()

# def predictions(model):
#     y_predictions= model.predict(X_test)
#     y_test.reshape(-1,)
#     y_predictions.reshape(-1,)
#     y_predictions= np.argmax(y_predictions, axis=1)
#     y_test= np.argmax(y_test, axis=1)

#     L = 8
#     W = 8
#     fig, axes = plt.subplots(L, W, figsize = (20,20))
#     axes = axes.ravel() # 

#     for i in np.arange(0, L * W):  
#         axes[i].imshow(X_test[i])
#         axes[i].set_title("Predicted = {}\n Actual  = {}".format(classes[y_predictions[i]], classes[y_test[i]]))
#         axes[i].axis('off')

#     plt.subplots_adjust(wspace=1)
#     plt.show()
