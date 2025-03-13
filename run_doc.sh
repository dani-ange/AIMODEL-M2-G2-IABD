#!/bin/bash

# Script pour générer automatiquement la documentation avec Pydoc

echo "Génération de la documentation avec Pydoc..."

python -m pydoc -w Scripts.preprocessing
python -m pydoc -w Scripts.training
python -m pydoc -w Scripts.evaluation

mv *.html Docs/

echo "Documentation générée avec succès et déplacée dans le dossier Docs/."
