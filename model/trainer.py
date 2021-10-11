import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle
from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# Opening the files containing data
X = pickle.load(open("X.pickle", "rb"))
y = np.array(pickle.load(open("y.pickle", "rb")))

# normalizing data (a pixel goes from 0 to 255)
X = X/255.0

flag=1

if flag==0:
	# Building the model
	model = Sequential()
	# 3 convolutional layers
	model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.5))

	# 2 hidden layers
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation("relu"))
	model.add(Dropout(0.5))

	model.add(Dense(128))
	model.add(Activation("relu"))
	#model.add(BatchNormalization())
	model.add(Dropout(0.5))


	# The output layer with 3 neurons, for 3 classes
	model.add(Dense(3))
	model.add(Activation("softmax"))

if flag==1:
	""" https://towardsdatascience.com/creating-image-classification-model-with-bayesian-perspective-a90a5956b14e """
	model = Sequential([
    Conv2D(input_shape = X.shape[1:], filters=8, kernel_size=16, activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(units=3, activation='softmax')
])
model.summary()

# Add stop and if model starts to overfit
early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

# Compiling the model using some basic parameters
model.compile(loss="sparse_categorical_crossentropy",
				optimizer="adam",
				metrics=["accuracy"])


# Training the model, with 40 iterations
# validation_split corresponds to the percentage of images used for the validation phase compared to all the images
history = model.fit(X, y, batch_size=32, epochs=50, validation_split=0.15, callbacks=[early_stopping_monitor])

# Saving the model
model_json = model.to_json()
with open("model.json", "w") as json_file :
	json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")

model.save('CNN.model')

test_loss, test_acc = model.evaluate(X,  y, verbose=2)
print("test accuracy", test_acc)
print("test loss", test_loss)

# Printing a graph showing the accuracy changes during the training phase

plt.style.use('paperfig')
fig, ax = plt.subplots(1,2,figsize=(8,5))
#
ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('model accuracy')
ax[0].set_ylabel('accuracy')
ax[0].set_xlabel('epoch')
ax[0].legend(['train', 'validation'], loc=4)
#
ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('model loss')
ax[1].set_ylabel('loss')
ax[1].set_xlabel('epoch')
ax[1].legend(['train', 'validation'], loc=1)
#
plt.tight_layout()
plt.savefig('accuracy.png', format='png', dpi=200)
plt.show()
