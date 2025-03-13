import pandas as pd

class preprocessing():
    """
        cette classe permet d'effectuer toute les operations de 
        preprocessing sur l'ensemble de donnees.
    """
    def Types(df):
        type_colonne = df.dtypes
        return type_colonne
