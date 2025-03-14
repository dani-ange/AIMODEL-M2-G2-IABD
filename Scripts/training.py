"""
Script d'entraînement des modèles SVM et KNN pour la classification du diabète.
Optimisation des hyperparamètres avec GridSearchCV.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os


class Trainer:
    def __init__(self):
        """Charge les ensembles de données prétraités."""
        self.x_train = np.load('data/x_train.npy')
        self.y_train = np.load('data/y_train.npy')
        self.models = {}

    def train_svm(self):
        """Entraîne un modèle SVM avec un noyau linéaire."""
        svm_model = SVC(kernel='linear')
        svm_model.fit(self.x_train, self.y_train)
        self.models['svm'] = svm_model

    def train_knn(self):
        """Entraîne un modèle KNN avec recherche d'hyperparamètres."""
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.x_train, self.y_train)
        self.models['knn'] = grid_search.best_estimator_
        print(f"Meilleurs paramètres KNN: {grid_search.best_params_}")

    def save_models(self):
        os.makedirs('model', exist_ok=True)
        """Sauvegarde les modèles entraînés."""
        for model_name, model in self.models.items():
            joblib.dump(model, f'model/{model_name}_model.pkl')
        print("Modèles sauvegardés avec succès.")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_svm()
    trainer.train_knn()
    trainer.save_models()
