# Maissa Itchir
# Samia Bouchaal

import utils
import pandas as pd
import numpy as np
import math

def getPrior(df):
    """
    Cette fonction calcule la probabilite à priori de la classe $1$ ainsi que l'intervalle de confiance

    Parameters
    ----------
    df : DataFrame etudié

    Results
    -------
    Dict['estimation','min5pourcent','max5pourcent']
    """

    moyenne=df.loc[:,"target"].mean()
    ecart_type=df.loc[:,"target"].std()

    marge_erreur=1.96*(ecart_type/math.sqrt(df.shape[0])
    min=moyenne-marge_erreur
    max=moyenne+marge_erreur

    return {'estimation' : moyenne, 'min5pourcent' : min , 'max5pourcent': max}
