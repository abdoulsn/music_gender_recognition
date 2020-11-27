import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model.h5"
LEARNING_RATE = 0.0001


def load_data(data_path):
    """Chargé les données via csv .

    : param data_path (str): chemin vers le csv
     : retourne X un pandas dataframe: les predicteures
     : return y un pandas: la cibles

    """
    with open(data_path, "r") as fp:
        data = pd.read_csv(fp)

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    print("Base totale chargé!")
    return X, y


def echantillons(data_path, test_size=0.2, validation_size=0.2):
    """Creation train, validation et test.

     : param data_path (str): chemin vers le fichier csv contenant des données
     : param test_size (flaot): pourcentage du jeu de données utilisé pour les tests
     : param validation_size (float): Pourcentage de rame utilisé pour la validation croisée

     : return X_train (dataframe): Entrées pour la rame
     : return y_train (dataframe): Cibles pour le train
     : return X_validation (dataframe): Entrées pour l'ensemble de validation
     : return y_validation (dataframe): Cibles pour l'ensemble de validation
     : return X_test (dataframe): Entrées pour l'ensemble de test
     :return X_test (dataframe): Targets for the test set
    """

    # charge de la base
    X, y = load_data(data_path)

    # Extraction train, validation, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # gestion des axes
    # X_train = X_train[..., np.newaxis]
    # X_test = X_test[..., np.newaxis]
    # X_validation = X_validation[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test

if __name__ == "__main__":
    echantillons(data_path="", test_size=0.2, validation_size=0.2)
    