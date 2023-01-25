#!/usr/bin/env python3
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras import datasets, layers, models, losses
from keras import backend as K


def define_alexnet_model():
    model = models.Sequential()

    model.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation="bilinear", input_shape=(28,28,1)))

    model.add(layers.Conv2D(96, 11, strides=4, padding='same'))
    model.add(layers.Lambda(tf.nn.local_response_normalization))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(3, strides=2))

    model.add(layers.Conv2D(256, 5, strides=4, padding='same'))
    model.add(layers.Lambda(tf.nn.local_response_normalization))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(3, strides=2))

    model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(384, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(256, 3, strides=4, padding='same'))
    model.add(layers.Activation('relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    return model

model_alex = define_alexnet_model()
model_alex.load_weights('model_alex.h5')

# Precision (using keras backend)
def precision_metric(y_true, y_pred):
    threshold = 0.5  # Training threshold 0.5
    y_pred_y = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold), K.floatx())

    true_positives = K.sum(K.clip(y_true * y_pred, 0, 1))
    false_negatives = K.sum(K.clip(y_true * (1-y_pred), 0, 1))
    false_positives = K.sum(K.clip((1-y_true) * y_pred, 0, 1))
    true_negatives = K.sum(K.clip((1 - y_true) * (1-y_pred), 0, 1))

    precision = true_positives / (true_positives + false_positives + K.epsilon())
    return precision

# Recall (using keras backend)
def recall_metric(y_true, y_pred):
    threshold = 0.5 #Training threshold 0.5
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold), K.floatx())

    true_positives = K.sum(K.clip(y_true * y_pred, 0, 1))
    false_negatives = K.sum(K.clip(y_true * (1-y_pred), 0, 1))
    false_positives = K.sum(K.clip((1-y_true) * y_pred, 0, 1))
    true_negatives = K.sum(K.clip((1 - y_true) * (1-y_pred), 0, 1))

    recall = true_positives / (true_positives + false_negatives + K.epsilon())
    return recall

# F1-score (using keras backend)
def f1_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (recall+precision+K.epsilon()))
    return f1

def build_model():
    inp = keras.Input(shape=(28,28,1))
    x = keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(1,1),padding='SAME', 
                              activation='relu')(inp)
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(5,5), padding='SAME', activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inp, outputs=output)

    model.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy', f1_metric, recall_metric, precision_metric])
    return model, inp, output


model = keras.models.load_model('model_cnn',custom_objects={'f1_metric':f1_metric, 'recall_metric':recall_metric,'precision_metric':precision_metric})

def predict_digit(img):     
    #resize image to 28x28 pixels
    dim = (28,28)
    img = img.resize(dim)   
    #convert rgb to grayscale
    img = img.convert('L')    
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    return res

def crop_image(img_entree, coordonnees):
    crop_rectangle = coordonnees
    img_resized = img_entree.crop(crop_rectangle)
    probs = predict_digit(img_resized)
    print(np.argsort(probs)[::-1][:3])
    return probs

def decoupe_img(input_img, matrice_coordonnees, resp):
    for x in range(len(matrice_coordonnees)):
        for y in range(len(matrice_coordonnees[x])):
            if matrice_coordonnees[x][y][0] != 0 and matrice_coordonnees[x][y][1] != 0 and matrice_coordonnees[x][y][2] != 0: 
                left = matrice_coordonnees[x][y][0] - matrice_coordonnees[x][y][2] - 1
                upper = matrice_coordonnees[x][y][1] - matrice_coordonnees[x][y][2] - 1
                right = matrice_coordonnees[x][y][0] + matrice_coordonnees[x][y][2] + 1
                lower = matrice_coordonnees[x][y][1] + matrice_coordonnees[x][y][2] + 1
                coordonnees = (left, upper, right, lower)
                #crop de l'image
                probs = crop_image(input_img, coordonnees)
                resp["islands"].append({"row": x, "col": y, "digits_probabilities": probs.tolist()[1:9]})
    return resp
