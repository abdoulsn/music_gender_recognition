#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def build_model(*args, loss="", learning_rate=0.0001):
    """Contruire un modele SVM avec scikitlearn.

    :return model: TensorFlow model
    """

    pass

    return


def train(model, X_train, y_train, X_validation, y_validation):
    """Trains du modèle
     : param X_train (dataframe): Entrées pour
     : param y_train (dataframe): Cibles pour
     : param X_validation (dataframe): Entrées pour l'ensemble de validation
     : param y_validation (dataframe): Cibles pour l'ensemble de validation

    """

    pass

    return




def main():
    # generons les données
    X_train, y_train, X_validation, y_validation, X_test, y_test = echantillons(DATA_PATH)

    # SVM
    model = None

    # train network
    history = train(model, X_train, y_train, X_validation, y_validation)

    # précision / perte
    pass

    # evaluation sur les données de test
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("\nTest de loss: {}, test de accuracy: {}".format(test_loss, 100*test_acc))

    # Enregistrer le modéle
    pass

if __name__ == "__main__":
    main()
