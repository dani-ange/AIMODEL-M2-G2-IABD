"""
Script d'évaluation des modèles de classification SVM et KNN pour le diabète.
"""

import numpy as np
import joblib
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Evaluator:
    def __init__(self):
        """Charge les ensembles de test et les modèles sauvegardés."""
        self.x_test = np.load('data/x_test.npy')
        self.y_test = np.load('data/y_test.npy')
        self.models = {
            'svm': joblib.load('model/svm_model.pkl'),
            'knn': joblib.load('model/knn_model.pkl')
        }
        self.scores = {}

    def evaluate(self):
        """Évalue les performances des modèles avec les données de test."""
        for model_name, model in self.models.items():
            y_pred = model.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.scores[model_name] = accuracy

            print(f"\nModèle: {model_name}")
            print(f"Exactitude: {accuracy:.4f}")
            print("Rapport de classification:")
            print(classification_report(self.y_test, y_pred))
            print("Matrice de confusion:")
            print(confusion_matrix(self.y_test, y_pred))

        # Enregistrer les scores dans un fichier JSON
        with open("model/scores.json", "w") as f:
            json.dump(self.scores, f)
        print("\nScores enregistrés dans model/scores.json")

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.evaluate()
