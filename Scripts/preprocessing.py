"""
Module de prétraitement des données pour le modèle de classification du diabète.

Ce script charge les données, gère les valeurs manquantes, effectue l'analyse statistique, la normalisation et la séparation en ensembles d'entraînement et de test.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Preprocessing:
    def __init__(self, filepath):
        """
        Initialise la classe avec le chemin du fichier de données.

        :param filepath: Chemin du fichier CSV contenant les données
        """
        self.filepath = filepath
        self.data = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Charge les données à partir d'un fichier CSV."""
        self.data = pd.read_csv(self.filepath)
        print("Données chargées avec succès.")

    def handle_missing_values(self):
        """Gère les valeurs manquantes en les remplaçant par la médiane."""
        self.data.fillna(self.data.median(), inplace=True)
        print("Valeurs manquantes remplacées par la médiane.")

    def detect_outliers(self):
        """Détecte les valeurs aberrantes en utilisant l'IQR et les remplace par les limites."""
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        self.data = self.data.clip(lower=lower_bound, upper=upper_bound, axis=1)
        print("Valeurs aberrantes détectées et corrigées.")

    def normalize_and_split(self):
        """Effectue la normalisation et la séparation des données."""
        x = self.data.drop(['Outcome'], axis=1)
        y = self.data['Outcome']

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x_scaled, y, test_size=0.2, stratify=y, random_state=2
        )
        print("Données normalisées et divisées en ensembles d'entraînement et de test.")

    def save_preprocessed_data(self):
        """Enregistre les ensembles de données prétraités."""
        np.save('../data/x_train.npy', self.x_train)
        np.save('../data/x_test.npy', self.x_test)
        np.save('../data/y_train.npy', self.y_train)
        np.save('../data/y_test.npy', self.y_test)
        print("Données prétraitées enregistrées.")


if __name__ == "__main__":
    preprocessor = Preprocessing('../data/diabetes.csv')
    preprocessor.load_data()
    preprocessor.handle_missing_values()
    preprocessor.detect_outliers()
    preprocessor.normalize_and_split()
    preprocessor.save_preprocessed_data()
    print("Prétraitement complet terminé avec succès.")
