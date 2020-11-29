#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout,Input
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
from prepare_data import *

DATA_PATH = "../data_out/rawdata.csv"
SAVED_MODEL_PATH = "../data_out/modeles/nnet.h5"
EPOCHS = 300
PATIENCE = 20
LEARNING_RATE = 0.001


def build_model(input_shape, loss="categorical_crossentropy", learning_rate=0.0001):
    """Construction d'un réseau neuronal à l'aide de keras.

    : param input_shape (tuple): Forme du df représentant un data train.
    : param loss (str): fonction de perte à utiliser
    : param learning_rate (float):
    : modèle de retour: modèle TensorFlow
    """

    # LE réseau

    inp=Input(shape=(input_shape,))
    model = Dense(500,activation='relu')(inp)
    model = Dropout(0.3)(model)
    model = Dense(8000,activation='relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(4000,activation='relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(2000,activation='relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(1000,activation='relu')(model)
    model = Dense(500,activation='relu')(model)
    model = Dense(10,activation='softmax')(model)

    model = Model(inputs=inp,outputs=model)
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])

    # print summary du modèle
    model.summary()

    return model


def train(model, epochs, patience, x_train, y_train):
    """ apprentisage
    : param epochs (int): nbre d'itérations d'apprentissage
    : param patience (int): Nombre d'époques à attendre avant l'arrêt anticipé, s'il n'y a pas d'amélioration de la
        précision
    : param x_train (ndarray): Entrées pour la df X
    : param y_train (ndarray): Cibles pour la df Y
    : param X_validation (ndarray): Entrées pour l'ensemble de validation
    : param y_validation (ndarray): Cibles pour l'ensemble de validation

    : return history et model: historique d'entraînement
    """
    # earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.0001, patience=)

    # train model
    lr=tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy',factor=0.5,patience=3,verbose=1)
    es=tf.keras.callbacks.EarlyStopping(monitor='accuracy',patience=patience,verbose=1)
    history = model.fit(x_train, y_train, epochs=epochs,
                        validation_split = .05,
                        callbacks=[lr, es])

    return history, model


def plot_history(history):
    """Plots accuracy/loss pour training/validation
    :param history
    :return:
    """

    fig, axs = plt.subplots(2)
    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()


def run_model():
    # generate train, validation and test sets
    x_train, y_train, x_validation, y_validation = echantillons(DATA_PATH, test_size=0.1)
    y_train = y_train.astype('category')
    y_train = y_train.cat.codes
    y_validation = y_validation.astype('category')
    y_validation = y_validation.cat.codes
    # pour le sparse cat cross_ent
    y_train = tf.keras.utils.to_categorical(y_train,10,'int')
    y_validation = tf.keras.utils.to_categorical(y_validation,10,'int')

    # creation du réseau
    input_shape=x_train.shape[1]
    model = build_model(input_shape, learning_rate=LEARNING_RATE)
    # apprentisage network
    history, model = train(model, EPOCHS, PATIENCE, x_train, y_train) #, x_validation, y_validation)
    # plot accuracy/loss
    plot_history(history)

    # evaluation sur la df validation
    test_loss, test_acc = model.evaluate(x_validation, y_validation)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, test_acc))
    pred = model.predict(x_validation)
    preds = pd.DataFrame(pred)
    preds.to_csv("../data_out/preds.csv")
    # cm = confusion_matrix(y_validation,pred)
    # np.set_printoptions(precision=2)
    # print("La matrice de confusion")

    # enregistré le modèle
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    run_model()