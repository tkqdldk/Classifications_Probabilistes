# Maissa Itchir
# Samia Bouchaal

import utils
import pandas as pd
import numpy as np
import math




def getPrior(df):
    """
    Calcul la probabilite à priori de la classe $1$ ainsi que l'intervalle de confiance

    Parametres
    ---------
    df : pandas.dataframe
        le `pandas.dataframe` contenant les données

    Resultats
    -------
        Dict[estimation,min5pourcent,max5pourcent]
            Probabilite a priori et intervalle de confiance
    """

    moyenne=df.loc[:,"target"].mean() 
    ecart_type=df.loc[:,"target"].std() 

    me = 1.96*(ecart_type/math.sqrt(df.shape[0])) 

    minpourcent = moyenne-me

    maxpourcent = moyenne+me

    return {'estimation' : moyenne, 'min5pourcent' : minpourcent , 'max5pourcent': maxpourcent}

class APrioriClassifier(utils.AbstractClassifier):
    def __init__(self):
        pass
    
    
    
    def estimClass(self,attrs):
        """
        Cette fonction estime la classe à laquelle appartient un individu en fonction de ses attributs.

        Paramètres
        ----------
        attrs : dict
            Dictionnaire contenant les attributs de l'individu.

        Résultats
        -------
        int
            Estimation de la classe (1 pour malade, 0 pour non malade).
        """
        
        return 1
     
    
    def statsOnDF(self,df):
        """
        Cette fonction calcule les statistiques de qualité du Classifier en le confrontant à une base de donnée.

        Paramètres
        ----------
        df : pandas.DataFrame
            Le DataFrame contenant les données à évaluer.

        Résultats
        -------
        dict
            Un dictionnaire contenant les valeurs suivantes :
            - 'VP' : Vrai Positif. Le nombre d'individus avec target=1 et classe prédite=1.
            - 'VN' : Vrai Négatif. Le nombre d'individus avec target=0 et classe prédite=0.
            - 'FP' : Faux Positif. Le nombre d'individus avec target=0 et classe prédite=1.
            - 'FN' : Faux Négatif. Le nombre d'individus avec target=1 et classe prédite=0.
            - 'Précision' : La précision du Classifier.
            - 'Rappel' : Le rappel du Classifier.
        """
        dico = {'VP': 0, 'VN': 0, 'FP': 0, 'FN': 0 , 'Précision': 0 , 'Rappel': 0 }
        for t in df.itertuples():
            dic = t._asdict()  
            estimClass = self.estimClass(dic)  
            if dic['target'] == 1 and estimClass == 1:
                dico['VP'] += 1
            elif dic['target'] == 0 and estimClass == 0:
                dico['VN'] += 1
            elif dic['target'] == 0 and estimClass == 1:
                dico['FP'] += 1
            else:
                dico['FN'] += 1

        dico['Précision'] = dico['VP'] / (dico['VP'] + dico['FP'])
        dico['Rappel'] = dico['VP'] / (dico['VP'] + dico['FN'])

        return dico

def P2D_l(df,attr):
    """
    Calcul de la probabilité conditionnelle P(attribut | target)
  
    Parametes
    ---------
    df : pandas.dataframe
        le `pandas.dataframe` contenant les données
    att:  int
        le nom d'une colonne du DataFrame

    Results
    -------
        Dictionnaire de dictionnaire, contient P(attribut = a | target = t)
    
    """
    res = dict()
    for target in df['target'].unique():
        tmp = dict()
        for val_attr in df[attr].unique():
            tmp[val_attr] = 0
        res[target] = tmp
   
    group = df.groupby(["target", attr]).groups
    for t, val in group:
        res[t][val] = len(group[(t, val)])
   
    group = df.groupby(["target"])[attr].count()
    
    # calcul de la probabilité
    for target in res.keys():
        for val_attribut in res[target].keys():
            res[target][val_attribut] /= group[target]

    return res


def P2D_p(df, attr):
    """
    Calcul de la probabilité conditionnelle P(attribut | target) sous forme d'un dictionnaire.

    Paramètres
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    attr : str
        Le nom de l'attribut dont on veut calculer la probabilité conditionnelle.

    Résultats
    -------
    dict
        Un dictionnaire contenant les probabilités conditionnelles P(attribut = a | target = t).
    """
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
    """
    Classifieur 2D par maximum a posteriori à partir d'une seule colonne du dataframe.
    """
    def __init__(self, df, attr):
        """
        Initialise le classifieur. Crée un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target) à partir de P2D_p.
        
        Paramètres
        ----------
        df: pandas.DataFrame
            Le DataFrame contenant les données.
        attr: str
            Le nom de l'attribut dont on veut calculer la probabilité conditionnelle.
        
        Résultats
        -------
        None
        """    
        self.attr = attr
        self.dico = P2D_l(df, attr)

    def estimClass(self, attrs):
        """
        À partir d'un dictionnaire d'attributs, estime la classe par maximum a posteriori.
        
        Paramètres
        ----------
        attrs: dict
            Le dictionnaire nom-valeur des attributs.
        
        Résultats
        -------
        int
            La classe estimée.
        """
        val_attr = attrs[self.attr]
        classe = 0
        proba = 0
        for i in self.dico.keys():
            if self.dico[i][val_attr] >= proba :
                classe = i
                proba = self.dico[i][val_attr]
        return classe

class MAP2DClassifier(APrioriClassifier):
    """
    Classifieur 2D par maximum a posteriori à partir d'une seule colonne du dataframe.
    """
    def __init__(self, df, attr):
        """
        Initialise le classifieur. Crée un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target) à partir de P2D_p.
        
        Paramètres
        ----------
        df: pandas.DataFrame
            Le DataFrame contenant les données.
        attr: str
            Le nom de l'attribut dont on veut calculer la probabilité conditionnelle.
        
        Résultats
        -------
        None
        """
        self.attr = attr
        self.dico = P2D_p(df, attr)
       
    def estimClass(self, attrs):
        """
        À partir d'un dictionnaire d'attributs, estime la classe par maximum a posteriori.
        Paramètres
        ----------
        attrs: dict
            Le dictionnaire nom-valeur des attributs.
        
        Résultats
        -------
        int
            La classe estimée.
        """
        val_attr = attrs[self.attr]
        classe = 0
        proba = 0
        for i in self.dico[val_attr].keys():
            if self.dico[val_attr][i] >= proba :
                classe = i
                proba = self.dico[val_attr][i]
        return classe

#####
# Question 2.4 : comparaison
#####
# Nous préférons ... parce que ...
# et aussi parce que ...
#####


def nbParams(df, liste = None):
    """
    Calcule le nombre de paramètres nécessaires pour représenter les données du dataframe.
    
    Paramètres
    ----------
        df (pandas.DataFrame): Le dataframe contenant les données.
        liste (list, optional): Liste facultative de noms de colonnes. Si non spécifié, 
                                les colonnes du dataframe seront utilisées par défaut.
    Résultats
    -------
        None
    """
    taille = 8
    if liste is None:
        liste = list(df.columns)
    for attr in liste:
        taille *= np.unique(df[attr]).size
    print("{} variable(s) : {} octets".format(len(liste), taille))


def nbParamsIndep(df):
    """
    Calcule le nombre de paramètres nécessaires pour représenter les données du dataframe.
    
    Paramètres
    ----------
        df (pandas.DataFrame): Le dataframe contenant les données.
    
    Résultats
    -------
        None
    """
    taille = 0
    liste = list(df.columns)
    for attr in liste:
        taille += np.unique(df[attr]).size * 8
    print("{} variable(s) : {} octets".format(len(liste), taille))


#####
# Question 3.3 : Preuve
#####
# 
# 
#####


def drawNaiveBayes(df, root):
    """
    Construit un graphe orienté représentant naïve Bayes.
    
    Paramètres
    ----------
        df: Dataframe contenant les données.
        root: Le nom de la colonne du Dataframe utilisée comme racine.
    Résultats
    -------
    Une représentation graphique du modèle naïve Bayes.
    """
    chaine = ""

    attrs = list(df.columns); #recuperer tous les attributs  
    attrs.remove(root);

    for attr in attrs:
        chaine += root + "->" + attr + ";"
    return utils.drawGraph(chaine)




    