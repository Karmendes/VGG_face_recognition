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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.metrics import classification_report
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os


# %%
# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images("data"))
data = []
labels = []

for imagePath in imagePaths:
	# extract the class label from the filename, load the image and
	# resize it to be a fixed 32x32 pixels, ignoring aspect ratio
	label = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	try:
		image = cv2.resize(image, (224, 224))
		#image = np.expand_dims(image, axis=0)
	except:
		print("Imagem com erro",imagePath)

	# update the data and labels lists, respectively
	data.append(image)
	labels.append(label)


# %%
# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")


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
# set the callbacks
callback = [
    ModelCheckpoint(
        filepath='model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_acc',
        mode='max',
        save_best_only=True
    ),
    TensorBoard(
        log_dir='./logs',
        update_freq=30
    ),
]


# %%
# train the model
print("[INFO] training model...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=32),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // 32,
	epochs=240,
	callbacks=[callback]
)


# %%
print("[INFO] saving model...")
model.save('model.h5',save_format="h5")


