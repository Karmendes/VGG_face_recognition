# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse


# %%
# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images('Train')))
totalVal = len(list(paths.list_images('Test')))


# %%
# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=25,
	zoom_range=0.1,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.2,
	horizontal_flip=True,
	fill_mode="nearest")
# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
valAug = ImageDataGenerator()
# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation
# objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean


# %%
# initialize the training generator
trainGen = trainAug.flow_from_directory(
	'Train/',
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=True,
	batch_size=32)
# initialize the validation generator
valGen = valAug.flow_from_directory(
	'Test/',
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=32)


# %%
# load the ResNet-50 network, ensuring the head FC layer sets are left
# off
print("[INFO] preparing model...")
vggface = VGGFace(model='senet50',include_top=False, input_shape=(224, 224, 3))


# %%
# construct the head of the model that will be placed on top of the
# the base model
headModel = vggface.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(1024, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.3)(headModel)
headModel = Dense(28, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=vggface.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the training process
for layer in vggface.layers:
	layer.trainable = False


# %%
# compile the model
opt = Adam(lr=1e-4, decay=1e-4 / 20)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


# %%
# train the model
print("[INFO] training model...")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // 32,
	validation_data=valGen,
	validation_steps=totalVal // 32,
	epochs=228)


# %%
print("[INFO] saving model...")
model.save('model.h5',save_format="h5")


