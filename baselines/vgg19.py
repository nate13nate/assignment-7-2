# Import libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from keras.optimizers import RMSprop
from keras.preprocessing import image
from keras.datasets import cifar10

from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator


# Extract data and train and test dataset

(X_train,Y_train) , (X_test,Y_test) = cifar10.load_data()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Let's look into the dataset images

plt.figure(figsize = (16,16))
for i in range(100):
  plt.subplot(10,10,1+i)
  plt.axis('off')
  plt.imshow(X_train[i], cmap = 'gray')

# Training , Validating and Splitting trained and tested data

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X_train,Y_train,test_size=0.2)

from keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes = 10)
y_val = to_categorical(y_val, num_classes = 10)

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(X_test.shape)
print(Y_test.shape)

# train_datagen = ImageDataGenerator(
#     preprocessing_function = tf.keras.applications.vgg19.preprocess_input,
#     rotation_range=10,
#     zoom_range = 0.1,
#     width_shift_range = 0.1,
#     height_shift_range = 0.1,
#     shear_range = 0.1,
#     horizontal_flip = True,
# )
# train_datagen.fit(x_train)

val_datagen = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg19.preprocess_input)
val_datagen.fit(x_val)

from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_learning_rate=0.00001)

# We have used only 16 layers out of 19 layers in the CNN
from keras.applications import VGG19
vgg_model = VGG19(
    include_top=False,
    weights="imagenet",
    input_shape=(32,32,3),
)

vgg_model.summary()

model = tf.keras.Sequential()
model.add(vgg_model)
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.summary()

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9)
model.compile(optimizer= optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

history = model.fit(
    x=x_train,
    y=y_train,
    validation_data = val_datagen.flow(x_val,y_val, batch_size = 128),
    epochs = 25,
    verbose = 1,
    callbacks = [learning_rate_reduction]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# plt.figure()
# plt.plot(acc,color = 'purple',label = 'Training Acuracy')
# plt.plot(val_acc,color = 'blue',label = 'Validation Accuracy')
# plt.legend()

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.figure()
# plt.plot(loss,color = 'green',label = 'Training Loss')
# plt.plot(val_loss,color = 'red',label = 'Validation Loss')
# plt.legend()

# x_test = tf.keras.applications.vgg19.preprocess_input(X_test)
# y_pred = model.predict_classes(x_test)
# y_pred[:10]

# from sklearn.metrics import confusion_matrix, accuracy_score
# print('Testing Accuarcy : ', accuracy_score(Y_test, y_pred))

# cm = confusion_matrix(Y_test, y_pred)
# cm

# import itertools
# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Greens):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=30)
#     plt.yticks(tick_marks, classes)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     #print(cm)

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#             horizontalalignment="center",
#             color="white" if cm[i, j] > thresh else "black")

#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

# plt.figure(figsize=(8,8))
# plot_confusion_matrix(cm,classes)

model.save('./baselines/models/vgg19.keras')

import pickle
with open('./baselines/history/vgg19', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
