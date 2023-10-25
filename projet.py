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



#VP : vrai positif . Le nombre d'individus avec target=1 et classe prévue=1
    #On a suppose que la personne est malade et elle est vraiment malade
#VN : vrai négatif . Le nombre d'individus avec target=0 et classe prévue=0
    #On a suppose que la personne n'est pas malde et vraiment elle est pas malade 
#FP : faux positif . Le nombre d'individus avec target=0 et classe prévue=1
    #On a suppose que la personne est malade mais en vri elle est pas malade 
#FN : faux négatif . Le nombre d'individus avec target=1 et classe prévue=0
    #On a suppose que la personne n'est pas malade mais en vrai elle est malade



class AprioriClassifier(utils.AbstractClassifier):
    def __init__(self):
        pass
    
    #La methode estimClass prends un dictionnaire d'attributs qui permettent de decrire un individu
    # Elle retourne l'estimation de la classe à laquelle appartient un individu donne en fonction des ses attributs
    
    # si l'estimation retournee par 'getPriori' est superieure à 0.5 signifie que la classe malade est plus probable et donc on predit que l'individu est malade
    # si l'estimation est inferieure à 0.5 on predit que l'individu n'est pas malade 
    
    # Pour calculer l'estiamtion , on utilise la fonction getPriori qui prends en parametres un df (dans ce cas la je dois mettre les attributs du estimClass dans un df)
    
    def estimClass(self,attrs):
        """df_attrs = pd.DataFrame([attrs])
        Dict = getPrior(df_attrs)
        
        if Dict['estimation'] >= 0.5:
            return 1
        else:
            return 0"""
         # D'apres la logique de la methode suivante ce que j'ai fait dans cette fonction est correcte
         
        return 1
    
    # La methode statsOnDF prend en parametre un dataFrame
    # Elle doit retourner les resultats suivants
    """
    VP : vrai positif . Le nombre d'individus avec target=1 et classe prévue=1
    VN : vrai négatif . Le nombre d'individus avec target=0 et classe prévue=0
    FP : faux positif . Le nombre d'individus avec target=0 et classe prévue=1
    FN : faux négatif . Le nombre d'individus avec target=1 et classe prévue=0
    
    """
     
    # la precision = Vrais Positifs /  Vrais Positifs + faux negatifs 
    # le rappel = Vrais Positifs / elements pertinents 
    def statsOnDF(self,df):
        dico = {'VP': 0, 'VN': 0, 'FP': 0, 'FN': 0 , 'Précision': 0 , 'Rappel': 0 }
       
        for t in df.itertuples():                             # On parcourt le dataFRame 
            dic = t._asdict()                                 # A chaque iteration on recupere une ligne du dataFrame (c'est le dictionni)
            estimClass = self.estimClass(dic)                 # Ensuite on calcule l'estimation de chaque individu en appelant la methode estimClass qui prend en parametre la ligne recuperee du dataFrame
           
            if dic['target'] == 1 and estimClass == 1:
                dico['VP'] += 1                               # le nombre de Vrais Positifs
            elif dic['target'] == 0 and estimClass == 0:
                dico['VN'] += 1                               # le nombre de Vrais negatifs
            elif dic['target'] == 0 and estimClass == 1:
                dico['FP'] += 1                               # le nombre de faux Positifs
            else:
                dico['FN'] += 1                               # le nombre de faux negatifs
                   
        dico['Précision'] = dico['VP']/(dico['VP'] + dico['FP'])
        dico['Rappel'] = dico['VP']/ (dico['VP'] + dico['FN'])
       
        return dico


