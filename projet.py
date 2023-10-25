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

# Le resulatat de la fonction est un dictionnaire des dictionnaires
#  linstruction df['target'].unique() renvoie les valeurs uniques de la colonne target dans le dataFrame qui 
#  pour chaque paire (target, attribut)  on calcule la probabilite conditionnelle P(attribut=a∣target=t) 
def P2D_l(df,attr):
    resultat = dict()
    for target in df['target'].unique():  
        tmp = dict()                      # On initialise un dictionnaire temporire pour chaque valeur
         for val_attr in df[attr].unique():  # On parcourt les valeurs uniques de la colonne specifiee par 'attr'
            tmp[val_attr] = 0
        res[target] = tmp # le dictionnaire initiale contient la valeur du target ans le premier champs et le dictionnaire temporaire dans le 2 eme champs
        group = df.groupby(["target", attr]).groups # regrouper  les données en fonction des combinaisons de la variable cible 'target'attribut
    for t, val in group: # arcourt les combinaisons de la variable cible et de l'attribut
        res[t][val] = len(group[(t, val)])  #Compte le nombre d'occurrences pour cette combinaison et l'assigne au dictionnaire de résultats res
   
    group = df.groupby(["target"])[attr].count()   # Compte le nombre d'occurrences de chaque valeur de l'attribut pour chaque classe de la variable cible
    
    
    for target in res.keys():  # Parcourt les clés (valeurs de l'attribut) dans le dictionnaire associé à la valeur de la variable cible 'target'
        for val_attribut in res[target].keys():
            res[target][val_attribut] /= group[target] #  Calcule la probabilité conditionnelle en divisant le nombre d'occurrences par le nombre total d'occurrences de cette classe de la variable cible

    return res



# calcule les probabilités conditionnelles pour chaque combinaison de l'attribut et de la variable cible. 
  #  Le résultat est renvoyé sous forme d'un dictionnaire de dictionnaires.
    
def P2D_p(df, attr):
    res = dict()
    for val_attr in df[attr].unique():
        tmp = dict()
        for val_targ in df['target'].unique():
            tmp[val_targ] = 0
        res[val_attr] = tmp
       
    group = df.groupby([attr, "target"]).groups
    for t, val in group:
        res[t][val] = len(group[(t, val)])

    group = df.groupby([attr])['target'].count()
   
    for val_attr in res.keys():
        for val_targ in res[val_attr].keys():
            res[val_attr][val_targ] /= group[val_attr]
    return res  




class ML2DClassifier(APrioriClassifier):
    def __init__(self, df, attr):
        self.attr = attr
        self.dico = P2D_l(df, attr)

    def estimClass(self, attrs):
        val_attr = attrs[self.attr]
        classe = 0
        proba = 0
        for i in self.dico.keys():
            if self.dico[i][val_attr] >= proba :
                classe = i
                proba = self.dico[i][val_attr]
        return classe
    
    
class MAP2DClassifier(APrioriClassifier):
    def __init__(self, df, attr):
        self.attr = attr
        self.dico = P2D_p(df, attr)
       
    def estimClass(self, attrs):
        val_attr = attrs[self.attr]
        classe = 0
        proba = 0
        for i in self.dico[val_attr].keys():
            if self.dico[val_attr][i] >= proba :
                classe = i
                proba = self.dico[val_attr][i]
        return classe
