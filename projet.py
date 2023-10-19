# Maissa Itchir
# Samia Bouchaal

import utils
import pandas as pd
import numpy as np
import math
import utils



def getPrior(df):
    """
    Cette fonction calcule la probabilite à priori de la classe $1$ ainsi que l'intervalle de confiance

    Parameters
    ----------
        df : DataFrame
            Base de donnees utilisee

    Results
    -------
        Dict[estimation,min5pourcent,max5pourcent]
            Probabilite a priori et intervalle de confiance
    """

    moyenne=df.loc[:,"target"].mean() #calcule la moyenne des valeurs de la colonne target
    ecart_type=df.loc[:,"target"].std() # calcule l'ecart-type

    me = 1.96*(ecart_type/math.sqrt(df.shape[0])) #marge d'erreur

    minpourcent = moyenne-me

    maxpourcent = moyenne+me

    return {'estimation' : moyenne, 'min5pourcent' : minpourcent , 'max5pourcent': maxpourcent}



