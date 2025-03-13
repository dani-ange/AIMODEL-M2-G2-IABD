#!/bin/bash

# Script pour générer automatiquement la documentation avec Pydoc

echo "Génération de la documentation avec Pydoc..."

pydoc -w script.preprocessing
pydoc -w script.train
pydoc -w script.evaluation

mv *.html doc/

echo "Documentation générée avec succès et déplacée dans le dossier doc/."