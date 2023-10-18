# Maissa Itchir
# Samia Bouchaal

import utils
import pandas as pd
import numpy as np

def getPrior(df):
    """
    Cette fonction calcule la probabilite à priori de la classe $1$ ainsi que l'intervalle de confiance

    Parameters
    ----------
    df : DataFrame

    Results
    -------
    Dict[estimation,
    """
    moyenne=df.loc[:,"target"].mean()

    return moyenne

getPrior(train.csv)
