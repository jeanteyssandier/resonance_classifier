"""
    Find the resonant state of a system using a convolutional neural network image classification algorithm.
    Based on:
    https://towardsdatascience.com/all-the-steps-to-build-your-first-image-classifier-with-code-cf244b015799
"""

import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

CATEGORIES = ["lib0", "lib180", "nothing"]

def prepare(file):
    IMG_SIZE = 128
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("CNN.model")

nr=5 # Number of rows
nf=3 # Number of classes

fig, ax = plt.subplots(nr, nf, figsize=(5*nf,3*nr))
j0 = j1 = jn = jt = 0
ax[0,0].set_title('Libration around 0')
ax[0,1].set_title('Libration around 180')
ax[0,2].set_title('No libration')
while jt<nr*nf:
    j = np.random.randint(100,350)
    k = np.random.randint(1,3)
    image_file = "run"+str(j)+"_"+str(k)+".png"
    image = prepare(image_file)
    prediction = model.predict([image])
    prediction = list(prediction[0])
    print(prediction)
    cat = CATEGORIES[prediction.index(max(prediction))]
    if cat == "lib0" and j0<nr:
        ax[j0,0].imshow(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE), cmap="gray", aspect='auto')
        ax[j0,0].xaxis.set_visible(False)
        ax[j0,0].yaxis.set_visible(False)
        j0+=1
        jt+=1
    if cat == "lib180" and j1<nr:
        ax[j1,1].imshow(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE), cmap="gray", aspect='auto')
        ax[j1,1].xaxis.set_visible(False)
        ax[j1,1].yaxis.set_visible(False)
        j1+=1
        jt+=1
    if cat == "nothing" and jn<nr:
        ax[jn,2].imshow(cv2.imread(image_file, cv2.IMREAD_GRAYSCALE), cmap="gray", aspect='auto')
        ax[jn,2].xaxis.set_visible(False)
        ax[jn,2].yaxis.set_visible(False)
        jn+=1
        jt+=1
    print(jt, cat)

plt.tight_layout
#plt.savefig('result.png', format='png', dpi=200, bbox_inches='tight', pad_inches = 0.1)
plt.show()




#image = "run401_1.png"
#image = prepare(image)
#prediction = model.predict([image])
#prediction = list(prediction[0])
#print(CATEGORIES[prediction.index(max(prediction))])
