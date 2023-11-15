# Maissa Itchir
# Samia Bouchaal

import utils
import pandas as pd
import numpy as np
import math
from scipy.stats import chi2_contingency

import matplotlib
import matplotlib.pyplot as plt


def getPrior(df):
    """
    Calcul la probabilite à priori de la classe $1$ ainsi que l'intervalle de confiance

    Parametres
    ---------
        df: pandas.dataframe
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
            attrs: Dict[str,valeur]
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
            df: pandas.DataFrame
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
        df: pandas.dataframe
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
                Retourne La classe estimée.
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
# Nous préférons MAP2DClassifier parce que sa précision est largement supérieure à celle de APrioriClassifier
# et aussi parce que même si aussi précis que ML2DClassifier, le rappel de ce dernier est  légèrement plus faible.
# Ainsi, MAP2DClassifier maximise les éléments pertinents parmis les 3 classifieurs et sa précision est similaires
# à celle de ML2DClassifier. Malgré cela, les résultats de ces classifieurs ne sont pas satisfaisants :
# il y'a au minimum environ 20% de faux négatifs.
#####


def nbParams(df, liste = None):
    """
    Calcule le nombre de paramètres nécessaires pour représenter les données du dataframe.
    
    Paramètres
    ----------
        df: pandas.DataFrame
            Le dataframe contenant les données.
        liste: list, optional
            Liste facultative de noms de colonnes. Si non spécifié, les colonnes du dataframe seront utilisées par défaut.
    Résultats
    -------
        affiche et retourne le nombre d'octets
    """
    taille = 8
    res = 0
    if liste is None:
        liste = list(df.columns)
    for attr in liste:
        taille *= np.unique(df[attr]).size
    print("{} variable(s) : {} octets".format(len(liste), taille))
    return taille 


def nbParamsIndep(df):
    """
    Calcule le nombre de paramètres nécessaires pour représenter les données du dataframe.
    
    Paramètres
    ----------
        df : pandas.DataFrame
            Le dataframe contenant les données
    
    Résultats
    -------
        affiche et retourne la valeur calculee en octets 
        
    """
    taille = 0
    liste = list(df.columns)
    for attr in liste:
        taille += np.unique(df[attr]).size * 8
    print("{} variable(s) : {} octets".format(len(liste), taille))
    return taille


#####
# Question 3.3.a : preuve
#####
#
# Soit A, B et C des variables aléatoires. On a A indépendant de C sachant B. Montrons que P(A,B,C) = P(A)*P(B|A)*P(C|B).
# On a : P((A,C)|B)=P(A|B)*P(C|B)
#                  =(P(A,B)/P(B))*(P(C,B)/P(B)) en utilisant un définition des probabilités conditionnelles
#
# Or : P(A,B,C)=P(B)*P((A,C)|B))
#              =P(B)*(P(A,B)/P(B))*(P(C,B)/P(B))
#              =P(A,B)*P(C,B)/P(B)
#              =(P(A)/P(A))*P(A,B)*P(C,B)/P(B)
#              =P(A)*(P(A,B)/P(A))*(P(C,B)/P(B))
#              =P(A)*P(B|A)*P(C|B) CQFD
# Donc, si A indépendant de C sachant B, on peut écrire la loi jointe P(A,B,C) = P(A)*P(B|A)*P(C|B).
#####

#####
# Question 3.3.b : complexité en indépendance partielle
#####
# Considérons que les 3 variables A,B et C ont 5 valeurs de taille 8 octets.
# Avec l'indépendance conditionnelles, il faut (5^3)*8=1000 octets pour représenter ces valeurs en mémoire.
# Tandis que sans l'indépendance, il faut (15*5+5)*8=640 octets pour les représenter en mémoire.
#####

#####
# Question 4.1 : Exemples
#####
# Voici notre proposition de code pour dessiner les graphes pour 5 variables complètement indépendantes:
# utils.drawGraphHorizontal("A;B;C;D;E")
# Voici notre proposition pour dessiner les graphes pour 5 variables sans indépendance:
# utils.drawGraphHorizontal("E;E->D;E->C;D->C;E->B;D->B;C->B;E->A;D->A;C->A;B->A")
# En effet, P(A,B,C,D,E)=P(A|B,C,D,E)*P(B|C,D,E)*P(C|D,E)*P(D|E)*P(E)
#####

#####
# Question 4.2 : naïve Bayes
#####
# Décomposons la vraisemblance P(attr1, attr2, attr3, ...|target):
# P(attr1, attr2, attr3, ...| target) = P(attr1|target)*P(attr2|target)*P(attr3|target)*...*P(attrn|target)
#
# Décomposons la distribution a posteriori P(target|attr1, attr2, attr3, ...) :
# P(target|attr1, attr2, attr3, ...) = P(attr1, attr2, attr3, ..., attrn| target)*P(target)/P(attr1, attr2, attr3, ..., attrn)
#                                    = P(attr1|target)*P(attr2|target)*P(attr3|target)*...*P(attrn|target)*P(target)/P(attr1, attr2, attr3, ..., attrn)
#####



def drawNaiveBayes(df, root):
    """
    Construit un graphe orienté représentant naïve Bayes.

    Paramètres
    ----------
        df: pandas.DataFrame
            Dataframe contenant les données.
        root: str
            Le nom de la colonne du Dataframe utilisée comme racine.
    
    Résultats
    -------
    Retourne le graphe naïve Bayes construit.
    """
    chaine = ""

    attrs = list(df.columns); #recuperer tous les attributs  
    attrs.remove(root);

    for attr in attrs:
        chaine += root + "->" + attr + ";"
    return utils.drawGraph(chaine)



def nbParamsNaiveBayes(df, root, attrs = None):
    """
    Affiche la taille mémoire de tables P(target), P(attr1|target),.., P(attrk|target)
    étant donné un dataframe df, la colonne racine col_pere et la liste [target,attr1,...,attrk],
    en supposant qu'un float est représenté sur 8 octets.
    
    Paramètres
    ----------
        df: pandas.DataFrame
            Dataframe contenant les données.
        root: str
            Le nom de la colonne du Dataframe utilisée comme racine.
        attrs: Liste
            Liste contenant les colonnes prises en considération pour le calcul.
    
    Résultats
    -------
        Affiche et retourne le nombre d'octets nécessaires pour représenter les tables
    """
    taille = np.unique(df[root]).size
    size = taille*8
   
    if attrs is None:
        attrs = list(df.columns)

    for attr in attrs:
        if attr == root:
            continue
        tmp = (taille * np.unique(df[attr]).size) * 8
        size += tmp
   
    print("{} variable(s) : {} octets".format(len(attrs), size))
    return size


class MLNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par maximum de vraissemblance utilisant le modèle naïve Bayes.
    """
    def __init__(self, df):
        """
        Initialise le classifieur. Crée un dictionnarie où la clé est le nom de
        chaque attribut et la valeur est un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target).
        
        Paramètres
        ----------
            df: pandas.DataFrame
                Dataframe contenant les données.
        """
        self.df = df
        self.dico_P2D_l = {}
        attrs = list(df.columns)
        attrs.remove("target")
        for attr in attrs:
            self.dico_P2D_l[attr] = P2D_l(df, attr)
   
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs,on estime la classe.        
        
        Paramètres
        ----------
            attrs: Dict[str,valeur]
                dictionnaire nom-valeur des attributs

        Résultats
        ------- 
            retourne la classe estimée
        """
        vraissemblance = self.estimProbas(attrs)
        classe = 0
        proba = 0.0
        for i in vraissemblance.keys():
            if vraissemblance[i] >= proba :
                classe = i
                proba = vraissemblance[i]
        return classe
       
    def estimProbas(self, attrs):
        """
        Calcule la vraisemblance par naïve Bayes : P(attr1, ..., attrk | target).

        Paramètres
        ----------
             attrs: Dict[str,valeur]
                dictionnaire nom-valeur des attributs

        Résultats
        -------
            retourne un dictionnaire contenant les probabilités de vraisemblance calculées pour chaque classe de la colonne 'target'
        """    
        val_target = self.df['target'].unique() #target [1,0]
        dico_proba = dict.fromkeys(val_target, 1) # dico_proba = {1:1 , 0:1}
       
        for attr in self.dico_P2D_l.keys() :
            dico_tmp  =  self.dico_P2D_l[attr]
            for key_target in dico_proba.keys():
                """
                Si la clé n'existe pas dans dico_tmp
                Alors on fait une anticipé en renvoyant {1:0.0, 0:0.0}
                """
                if attrs[attr] not in dico_tmp[key_target]:  
                    res = dict.fromkeys(val_target, 0.0)
                    return res
                dico_proba[key_target] *= dico_tmp[key_target][attrs[attr]]
        return dico_proba



class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par le maximum a posteriori en utilisant le modèle naïve Bayes.
    """
    def __init__(self, df):
        """
        Initialise le classifieur. Crée un dictionnarie où la clé est le nom de
        chaque attribut et la valeur est un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target). Cree aussi un
        dictionnaire avec les probabilités de target.
        Paramètres
        ----------
            df: pandas.DataFrame
                Dataframe contenant les données.
        """
        self.df = df
        self.dico_P2D_l = {}
        tab_col = list(df.columns.values)
        tab_col.remove("target")
        for attr in tab_col:
            self.dico_P2D_l[attr] = P2D_l(df, attr)
   
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs,on estime la classe.         
        Paramètres
        ----------
            attrs: Dict[str,valeur]
                dictionnaire nom-valeur des attributs
        -------
            retourne la classe estimée
        """
        vraissemblance = self.estimProbas(attrs)
        classe = 0
        proba = 0.0
        for i in vraissemblance.keys():
            if vraissemblance[i] >= proba :
                classe = i
                proba = vraissemblance[i]
        return classe
       

    def estimProbas(self, attrs):
        """
        Calcule la probabilité à posteriori par naïve Bayes : P(target | attr1, ..., attrk).
        Paramètres
        ----------
            attrs: Dict[str,valeur]
                dictionnaire nom-valeur des attributs
        -------
             Retourne le dictionnaire de probabilités à posteriori
        """    
        val_target = self.df['target'].unique() #target [1,0]
        df = self.df.target.value_counts()/self.df.target.count()
        dico_proba = df.to_dict() #convertir le df en dictionnaire dico_proba = {1:xxx, 0:xxx}
       
        for attr in self.dico_P2D_l.keys() :
            dico_tmp  =  self.dico_P2D_l[attr]
            for key_target in dico_proba.keys():
                """
                Si la clé n'existe pas dans dico_tmp
                Alors on fait une anticipé en renvoyant {1:0.0, 0:0.0}
                """
                if attrs[attr] not in dico_tmp[key_target]:  
                    res = dict.fromkeys(val_target, 0.0)
                    return res
                dico_proba[key_target] *= dico_tmp[key_target][attrs[attr]]
       
        somme_proba = sum(dico_proba.values()) # contient la somme des probas calculées
       
        for prob in dico_proba.keys() :
            dico_proba[prob] /= somme_proba
       
        return dico_proba



def isIndepFromTarget(df,attr,x) :
    """
    Vérifie si attr est indépendant de target au seuil de x%.
    Paramètres
    ----------
        df: pandas.DataFrame
                Dataframe contenant les données.
        attrs: Dict[str,valeur]
                dictionnaire nom-valeur des attributs
        x: int
            seuil de confiance.

    Résultats
    -------
        Retourne True si attr est indépendant de target, False sinon.
    """
    liste = pd.crosstab(df[attr], df.target) #transformer le df en liste
    _,p,_,_ = chi2_contingency(liste) # calculer le p_value
    if p < x : # on rejette l'hypothese si p_value est inferieur au seuil de x%
        return False
    return True



class ReducedMLNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par maximum de vraissemblance utilisant le modèle naïve Bayes reduit. 
    """   
    def __init__(self, df, x):
        """
        Initialise le classifieur. Crée un dictionnaire où la clé est le nom de
        chaque attribut et la valeur est un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target).
        Paramètres
        ----------
            df: pandas.DataFrame
                Dataframe contenant les données.
            x: int
                seuil de confiance.
        """
        self.df = df
        self.dico_P2D_l = {}
        attrs = list(df.columns) # recupere tous les attributs
        attrs.remove("target")
        for attr in attrs:
            if not isIndepFromTarget(df, attr, x):
                self.dico_P2D_l[attr] = P2D_l(df, attr)
   
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs, on estime la classe.        
        Paramètres
        ----------
            attrs: Dict[str,valeur]
                dictionnaire nom-valeur des attributs
        Résultats
        -------
            Retourne la classe estimée
        """
        vraissemblance = self.estimProbas(attrs)
        classe = 0
        proba = 0.0
        for i in vraissemblance.keys():
            if vraissemblance[i] > proba :
                classe = i
                proba = vraissemblance[i]
        return classe
       
    def estimProbas(self, attrs):
        """
        Calcule la vraisemblance par naïve Bayes : P(attr1, ..., attrk | target).
        Paramètres
        ----------
            attrs: Dict[str,valeur]
                dictionnaire nom-valeur des attributs
        Résultats
        -------
            Retourne un dictionnaire contenant les probabilités de vraisemblance pour chaque classe de la colonne 'target'.
        """    
        val_target = self.df['target'].unique() #target [1,0]
        dico_proba = dict.fromkeys(val_target, 1) # dico_proba = {1:1 , 0:1}
       
        for attr in self.dico_P2D_l.keys() :
            dico_tmp  =  self.dico_P2D_l[attr]
            for key_target in dico_proba.keys():
                """
                Si la clé n'existe pas dans dico_tmp
                Alors on fait une anticipé en renvoyant {1:0.0, 0:0.0}
                """
                if attrs[attr] not in dico_tmp[key_target]:  
                    res = dict.fromkeys(val_target, 0.0)
                    return res
                dico_proba[key_target] *= dico_tmp[key_target][attrs[attr]]
        return dico_proba
   
    def draw(self) :
        """
        Construit un graphe orienté représentant le modèle naïve Bayes réduit.
        """
        chaine = ""
        for attr in self.dico_P2D_l.keys():
            chaine += "target" + "->" + attr + ";"
        return utils.drawGraph(chaine)


class ReducedMAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par le maximum a posteriori en utilisant le modèle naïve Bayes.
    """
    def __init__(self, df, x):
        """
        Initialise le classifieur. Crée un dictionnaire où la clé est le nom de
        chaque attribut et la valeur est un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target). Crée aussi un
        dictionnaire avec les probabilités de target.
        Paramètres
        ----------
            df: pandas.DataFrame
                Dataframe contenant les données.
            x: int
                seuil de confiance.
        """
        self.df = df
        self.dico_P2D_l = {}
        tab_col = list(df.columns.values)
        tab_col.remove("target")
        for attr in tab_col:
            if not isIndepFromTarget(df, attr, x):
                self.dico_P2D_l[attr] = P2D_l(df, attr)
   
    def estimClass(self, attrs):
        """
        À partir d'un dictionnaire d'attributs, on estime la classe.        
        Paramètres
        ----------
            attrs: Dict[str,valeur]
                dictionnaire nom-valeur des attributs
        Résultats
        -------
            retourne la classe estimée
        """
        vraissemblance = self.estimProbas(attrs)
        classe = 0
        proba = 0.0
        for i in vraissemblance.keys():
            if vraissemblance[i] > proba :
                classe = i
                proba = vraissemblance[i]
        return classe
       

    def estimProbas(self, attrs):
        """
        Calcule la probabilité à posteriori par naïve Bayes : P(target | attr1, ..., attrk).
        Paramètres
        ----------
            attrs: Dict[str,valeur]
                dictionnaire nom-valeur des attributs
        Résultats
        -------
            Rerourne dictionnaire contenant les probabilités à posteriori pour chaque classe de la colonne 'target'.
        """    
        val_target = self.df['target'].unique() #target [1,0]
        df = self.df.target.value_counts()/self.df.target.count()
        dico_proba = df.to_dict() #convertir le df en dictionnaire dico_proba = {1:xxx, 0:xxx}
       
        for attr in self.dico_P2D_l.keys() :
            dico_tmp  =  self.dico_P2D_l[attr]
            for key_target in dico_proba.keys():
                """
                Si la clé n'existe pas dans dico_tmp
                Alors on fait une anticipé en renvoyant {1:0.0, 0:0.0}
                """
                if attrs[attr] not in dico_tmp[key_target]:  
                    res = dict.fromkeys(val_target, 0.0)
                    return res
                dico_proba[key_target] *= dico_tmp[key_target][attrs[attr]]
       
        somme_proba = sum(dico_proba.values()) # contient la somme des probas calculées
       
        for prob in dico_proba.keys() :
            dico_proba[prob] /= somme_proba
       
        return dico_proba
   
    def draw(self) :
        """
        Construit un graphe orienté représentant le modèle naïve Bayes réduit.
        """
        chaine = ""
        for attr in self.dico_P2D_l.keys():
            chaine += "target" + "->" + attr + ";"
        return utils.drawGraph(chaine)
           
#####
# Question 6.1
#####
# Dans une représentation graphique des points (précision, rappel), le points idéal serait en (1,1), ce qui correspond
# à une précision et un rappel égaux à 1 (pas de faux négatifs et pas de faux positifs).
# 
# Por comparer les différents classifieurs de notre projet dans cette représentations graphique, on peut observer quels
# points (précision, rappel) de nos classifeurs se rapprochent du point idéal. On peut utiliser la distance euclidienne
# pour cela ou éventuellement la distance de manhattan, en implémentant des poids pour donner plus d'importance à l'abscisse
# (précision) ou à l'ordonnée (rappel) en fonction de ce que l'on veut minimiser en priorité le taux de faux positifs
# ou le taux de faux négatifs.
#####


def mapClassifiers(dic, df):
    """
    Représente graphiquement les classifieurs dans l'espace (précision, rappel).
    Paramètres
        ----------
        dic: 
            Dictionnaire {nom: instance de classifier}.
        df: pandas.DataFrame
                Dataframe contenant les données.
    """
    precision = np.empty(len(dic))
    rappel = np.empty(len(dic))
    
    for i, nom in enumerate(dic):
         dico_stats = dic[nom].statsOnDF(df)
         precision[i] = dico_stats["Précision"]
         rappel[i] = dico_stats["Rappel"]
    
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Précision")
    ax.set_ylabel("Rappel")
    ax.scatter(precision, rappel, marker = 'x', c = 'red') 
    
    for i, nom in enumerate(dic):
        ax.annotate(nom, (precision[i], rappel[i]))
    
    plt.show()

#####
# Question 6.3: Conclusion
#####
# D'après les représentations graphiques obtenues le classifieur APrioriClassifier a le meilleur rappel ( à 1) mais la
# pire précision sur les deux bases.
#
# Les classifieurs 2D par maximum de vraisemblance et par maximum à posteriori sont consistents sur les ensemble d'
# apprentissage et de test ( précision et rappel à environ 85%)
#
# On observe ques les versions réduites des classifieurs MAPNaiveBayes et MLNaiveBayes réduisent tous deux le rappel
# et change légèrement la précision des classifieurs originaux (le premier a une précision plus faible tandis que le
# second a une précision plus elevée).
# Ces 4 classifieurs sont plus précis et ont un meilleur rappel sur la base de données d'apprentissage que les
# classifieurs 2D par maximum de vraisemblance et par maximum à posteriori mais ont des rappels bien plus inférieurs
# aux classifieurs ML2DClassifier et MAP2DClassifier sur la base de données de test (en dessous de 0.4). Le taux de faux
# négatifs donnés sur l'ensemble de tests résultant des ces classifieurs est beaucoup trop élevé.
#
# On conclue que le modèle naïf bayésien est bien plus performant en apprentissage que sur les données de test et que la
# consistence sur les deux bases des classifieurs 2D est plus fiable.
#####




def MutualInformation(df, x, y):

    """
    Calcule l'information mutuelle entre les colonnes x et y du dataframe.
    Paramètres
    ----------
        df: pandas.DataFrame
            Dataframe contenant les données.. 
        x: str
            nom d'une colonne du dataframe.
        y: str
            nom d'une colonne du dataframe.
    Résultats
    -------
        Retourne l'information mutuelle qui est calculée en multipliant cette matrice par la matrice P(x, y) et en sommant le résultat

    """
    list_x = np.unique(df[x].values) # Valeurs possibles de x.
    list_y = np.unique(df[y].values) # Valeurs possibles de y.
    
    dico_x = {list_x[i]: i for i in range(list_x.size)} 
    #un dictionnaire associant chaque valeur a leur indice en list_x.
    
    dico_y = {list_y[i]: i for i in range(list_y.size)} 
    #un dictionnaire associant chaque valeur a leur indice en list_y.
    
    mat_xy = np.zeros((list_x.size, list_y.size), dtype = int)
    #matrice des valeurs P(x,y)
    
    group = df.groupby([x, y]).groups
    
    for i, j in group:
        mat_xy[dico_x[i], dico_y[j]] = len(group[(i, j)]) 
    
    mat_xy = mat_xy / mat_xy.sum()
    
    mat_x = mat_xy.sum(1)
    #matrice des P(x)
    mat_y = mat_xy.sum(0)
    #matrice des P(y)
    mat_px_py = np.dot(mat_x.reshape((mat_x.size, 1)),mat_y.reshape((1, mat_y.size))) 
    #matrice des P(x)P(y)
    
    mat_res = mat_xy / mat_px_py
    mat_res[mat_res == 0] = 1
    #pour éviter des problèmes avec le log de zero
    mat_res = np.log2(mat_res)
    mat_res *= mat_xy
    
    return mat_res.sum()


def ConditionalMutualInformation(df,x,y,z):
    """
    Calcule l'information mutuelle conditionnelle entre les colonnes x et y du dataframe en considerant les deux comme dependantes de la colonne z.
    
    Paramètres
    ----------
        df: pandas.DataFrame
            Dataframe contenant les données.
        x: str
            nom d'une colonne du dataframe.
        y: str
            nom d'une colonne du dataframe.
    Résultats
    -------
        Retourne l'information mutuelle conditionnelle qui est calculée en multipliant cette matrice par la matrice P(x,y,z) et en sommant le résultat

    
    """
    list_x = np.unique(df[x].values) # Valeurs possibles de x.
    list_y = np.unique(df[y].values) # Valeurs possibles de y.
    list_z = np.unique(df[z].values) # Valeurs possibles de z.
    
    dico_x = {list_x[i]: i for i in range(list_x.size)} 
    #un dictionnaire associant chaque valeur a leur indice en list_x.
    
    dico_y = {list_y[i]: i for i in range(list_y.size)} 
    #un dictionnaire associant chaque valeur a leur indice en list_y.
    
    dico_z = {list_z[i]: i for i in range(list_z.size)} 
    #un dictionnaire associant chaque valeur a leur indice en list_z.
    
    mat_xyz = np.zeros((list_x.size, list_y.size, list_z.size), dtype = int)
    #matrice des valeurs P(x,y,z)
    
    group = df.groupby([x, y, z]).groups
    
    for i, j, k in group:
        mat_xyz[dico_x[i], dico_y[j], dico_z[k]] = len(group[(i, j, k)]) 

    
    mat_xyz = mat_xyz / mat_xyz.sum()
    
    mat_xz = mat_xyz.sum(1)
    #matrice des P(x, z)
    
    mat_yz = mat_xyz.sum(0)
    #matrice des P(y, z)
    
    mat_z = mat_xz.sum(0)
    #matrice des P(z)
    
    mat_pxz_pyz = mat_xz.reshape((list_x.size, 1, list_z.size)) * mat_yz.reshape((1, list_y.size, list_z.size)) 
    #matrice des P(x, z)P(y, z)
    
    mat_pxz_pyz[mat_pxz_pyz == 0] = 1
    
    mat_pz_pxyz = mat_z.reshape((1, 1, list_z.size)) * mat_xyz
    #matrice des P(z)P(x, y, z)
    
    mat_res = mat_pz_pxyz / mat_pxz_pyz
    mat_res[mat_res == 0] = 1
    #pour éviter des problèmes avec le log de zero
    mat_res = np.log2(mat_res)
    mat_res *= mat_xyz
    
    return mat_res.sum()

def MeanForSymetricWeights(a):   
    """
    Calcule la moyenne des poids pour une matrice a symétrique de diagonale nulle.
    La diagonale n'est pas prise en compte pour le calcul de la moyenne.
    Paramètres
    ----------
        a: numpy.ndarray
            Matrice symétrique de diagonale nulle.  
    Résultats
    -------
        Retourne la somme de tous les éléments de la matrice (à l'exception des éléments de la diagonale) 
        divisée par le nombre total d'éléments de la matrice moins le nombre d'éléments sur la diagonale
    """
    return a.sum()/(a.size - a.shape[0])

def SimplifyConditionalMutualInformationMatrix(a):
    """
    Annule toutes les valeurs plus petites que sa moyenne dans une matrice a 
    symétrique de diagonale nulle.
    Paramètres
    ----------
        a: numpy.ndarray
            Matrice symétrique de diagonale nulle.
    Résultats
    -------
        Toutes les valeurs de la matrice qui sont inférieures à la moyenne sont mises à zéro.
    """
    moy = MeanForSymetricWeights(a)
    a[a < moy] = 0


def Kruskal(df,a):
    """
    Applique l'algorithme de Kruskal au graphe dont les sommets sont les colonnes
    de df (sauf 'target') et dont la matrice d'adjacence ponderée est a.
    Les indices dans a doivent être dans le même ordre que ceux de df.keys().
    Paramètres
    ----------
        df: pandas.DataFrame
            Dataframe contenant les données. 
        a: numpy.ndarray
            Matrice symétrique de diagonale nulle. 
    Résultats
    -------
        Renvoie la représentation du graphe obtenu après l'application de l'algorithme de Kruskal
      
    """
    list_col = [x for x in df.keys() if x != "target"]
    list_arr = [(list_col[i], list_col[j], a[i, j]) for i in range(a.shape[0]) for j in range(i + 1, a.shape[0]) if a[i, j] != 0]
    
    list_arr.sort(key = lambda x: x[2], reverse = True)
    
    g = Graphe(list_col)
    
    for (u, v, poids) in list_arr:
        if g.find(u) != g.find(v):
            g.addArete(u, v, poids)
            g.union(u, v)
    return g.graphe    


class Graphe:
    """
    Structure de graphe pour l'algorithme de Kruskal. 
    """
  
    def __init__(self, sommets): 
        """
        Paramètres
        ----------
            sommets: Liste
                    liste de sommets
        """
        self.S = sommets 
        #Liste de sommets 
        self.graphe = [] 
        #liste representant les aretes du graphe 
        self.parent = {s : s for s in self.S}
        #dictionnaire ou la clé est un sommet et la valeur est sont père
        #dans la forêt utilisée par l'algorithme de kruskal.
        self.taille = {s : 1 for s in self.S}
        #dictionnaire des tailles des arbres dans la forêt
        

    def addArete(self, u, v, poids): 
        """
        Ajoute l'arete (u, v) avec poids.
        Paramètres
        ----------
            u: 
                le nom d'un sommet.
            v: 
                le nom d'un sommet.
            poids: 
                poids de l'arete entre les deux sommets.
        """
        self.graphe.append((u,v,poids)) 
  
    def find(self, u): 
        """
        Trouve la racine du sommet u dans la forêt utilisée par l'algorithme de
        kruskal. Avec compression de chemin.

        Paramètres
        ----------
            u: 
                le nom d'un sommet.
        Résultats
        -------
        """
        racine = u
        #recherche de la racine
        while racine != self.parent[racine]:
            racine = self.parent[u]
        #compression du chemin    
        while u != racine:
            v = self.parent[u]
            self.parent[u] = racine
            u = v
        return racine            
  

    def union(self, u, v):
        """
        Union ponderé des deux arbres contenant u et v. Doivent être dans deux
        arbres differents.
        Paramètres
        ----------
            u: 
                le nom d'un sommet.
            v: 
                le nom d'un sommet.
        """
        u_racine = self.find(u) 
        v_racine = self.find(v) 
  
        if self.taille[u_racine] < self.taille[v_racine]: 
            self.parent[u_racine] = v_racine 
            self.taille[v_racine] += self.taille[u_racine] 
        else: 
            self.parent[v_racine] = u_racine 
            self.taille[u_racine] += self.taille[v_racine] 

def ConnexSets(list_arcs):
    """
    Costruit une liste des composantes connexes du graphe dont la liste d'aretes
    est list_arcs.
    Paramètres
    ----------
        list_arcs: Liste
                liste de triplets de la forme (sommet1, sommet2, poids).
    Résultats
    -------
        Retourne la liste finale des composantes connexes
    """
    res = []
    for (u, v, _) in list_arcs:
        u_set = None
        v_set = None
        for s in res:
            if u in s:
                u_set = s
            if v in s:
                v_set = s
        if u_set is None and v_set is None:
            res.append({u, v})
        elif u_set is None:
            v_set.add(u)
        elif v_set is None:
            u_set.add(v)
        elif u_set != v_set:
            res.remove(u_set)
            v_set = v_set.union(u_set)
    return res


def OrientConnexSets(df, arcs, classe):
    """
    Utilise l'information mutuelle (entre chaque attribut et la classe) pour
    proposer pour chaque ensemble d'attributs connexes une racine et qui rend 
    la liste des arcs orientés.
    Paramètres
    ----------
    df: pandas.DataFrame
        Dataframe contenant les données. 
    arcs: Liste
        liste d'ensembles d'arcs connexes.
    classe: 
        colonne de réference dans le dataframe pour le calcul de l'information mutuelle.

    Résultats
    -------
        Retourne la liste des arcs orientés résultants.
    """
    arcs_copy = arcs.copy()
    list_sets = ConnexSets(arcs_copy)
    list_arbre = []
    for s in list_sets:
        col_max = ""
        i_max = -float("inf") 
        for col in s:
            i = MutualInformation(df, col, classe)
            if i > i_max:
                i_max = i
                col_max = col
        list_arbre += creeArbre(arcs_copy, col_max)
    return list_arbre


def creeArbre(arcs, racine): 
    """
    À partir d'une liste d'arcs et d'une racine, renvoie l'arbre orienté depuis
    cette racine. La liste arcs est modifié par cette fonction.
    Paramètres
    ----------
        arcs: Liste
            liste d'ensembles d'arcs connexes.
        racine: str
            nom d'un sommet.

    Résultats
    -------
        Retourne la liste des arcs orientés de l'arbre résultant
    """
    res = []
    file = [racine]
    while file != []:
        sommet = file.pop(0)
        arcs_copy = arcs.copy()
        for (u, v, poids) in arcs_copy:
            if sommet == u:
                res.append((u, v))
                arcs.remove((u, v, poids))
                file.append(v)
            elif sommet == v:
                res.append((v, u))
                arcs.remove((u, v, poids))
                file.append(u)
    return res


class MAPTANClassifier(APrioriClassifier):
    """
    Classifieur par le maximum a posteriori en utilisant le modèle TAN
    (tree-augmented naïve Bayes).
    """
    def __init__(self, df):
        """
        Initialise le classifieur. Crée la matrice de Conditional Mutual Information
        simplifiée, une liste des arcs retenus pour le modèle TAN, un dictionnarie
        pour les probabilités 2D et un dictionnaire pour les probabilités 3D.
        Cree aussi un dictionnaire avec les probabilités de target = 0 et target = 1.
        
        Paramètres
        ----------
        df: pandas.DataFrame
            Dataframe contenant les données. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
        """
        self.createCmis(df)
        arcs = Kruskal(df, self.cmis)
        self.liste_arcs = OrientConnexSets(df, arcs, "target")
        
        self.dico_P2D_l = {}
        self.dico_P3D_l = {}
        self.pTarget = {1: df["target"].mean()}
        self.pTarget[0] = 1 - self.pTarget[1] 
        
        tab_col = list(df.columns.values)
        tab_col.remove("target")

        for attr in tab_col:
            pere = self.is3D(attr)
            if pere is not False:
                self.dico_P3D_l[attr] = P3D_l(df, attr, pere)
            else:
                self.dico_P2D_l[attr] = P2D_l(df, attr)
    
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimée est faite par maximum à posteriori à partir de dico_res.
        
        Paramètres
        ----------
             attrs: Dict[str,valeur]
                le dictionnaire nom-valeur des attributs

        Résultats
        -------
            Retourne la classe 0 ou 1 estimée
        """
        dico_res = self.estimProbas(attrs)
        if dico_res[0] >= dico_res[1]:
            return 0
        return 1
        

    def estimProbas(self, attrs):
        """
        Calcule la probabilité a posteriori P(target | attr1, ..., attrk) par
        la méthode TAN (tree-augmented naïve Bayes).

        Paramètres
        ----------
             attrs: Dict[str,valeur]
                le dictionnaire nom-valeur des attributs
        """
        P_0 = self.pTarget[0]
        P_1 = self.pTarget[1]
        for key in self.dico_P2D_l:
            dico_p = self.dico_P2D_l[key]
            if attrs[key] in dico_p[0]:
                P_0 *= dico_p[0][attrs[key]]
                P_1 *= dico_p[1][attrs[key]]
            else:
                return {0: 0.0, 1: 0.0}
        
        for key in self.dico_P3D_l:
            proba = self.dico_P3D_l[key]
            P_0 *= proba.getProba(attrs[key], attrs[proba.pere], 0)
            P_1 *= proba.getProba(attrs[key], attrs[proba.pere], 1)
        
        if (P_0 + P_1) == 0 : 
            return {0: 0.0, 1: 0.0}
        
        P_0res = P_0 / (P_0 + P_1)
        P_1res = P_1 / (P_0 + P_1)
        return {0: P_0res, 1: P_1res}
    
    def draw(self):
        """
        Construit un graphe orienté représentant le modèle TAN.
        """
        res = ""
        for enfant in self.dico_P2D_l:
            res = res + "target" + "->" + enfant + ";"
        for enfant in self.dico_P3D_l:
            res = res + self.dico_P3D_l[enfant].pere + "->" + enfant + ";"
            res = res + "target" + "->" + enfant + ";"
        return utils.drawGraph(res[:-1])
    
    def createCmis(self, df):
        """
        Crée la matrice de Conditional Mutual Information simplifiée à partir du dataframe df.
        
        Paramètres
        ----------
        df: pandas.DataFrame
            Dataframe contenant les données. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
        """
        self.cmis = np.array([[0 if x == y else ConditionalMutualInformation(df, x, y, "target") 
                                 for x in df.keys() if x != "target"] for y in df.keys() if y != "target"])
        SimplifyConditionalMutualInformationMatrix(self.cmis)
        
    def is3D(self, attr):  
        """
        Détermine si l'attribut attr doit être représenté par une matrice 3D,
        c'est-à-dire s'il a un parent outre que "target" dans self.list_arcs.
        
        Paramètres
        ----------
            attr: str
                nom d'un attribut du dataframe.
        """
        for pere, fils in self.liste_arcs:
            if fils == attr:
                return pere
        return False



class P3D_l():
    """
    Classe pour le calcul des probabilités du type P(attr1 | attr2, target).
    """
    def __init__(self, df, attr1, attr2):
        """
        Paramètres
        ----------
        df: pandas.DataFrame
            Dataframe contenant les données. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
        attr1: str
            nom d'une colonne du dataframe.
        attr2: str
            nom d'une colonne du dataframe.
        """
        self.pere = attr2
        list_x = np.unique(df[attr1].values) 
        list_y = np.unique(df[attr2].values) 
         
        self.dico_x = {list_x[i]: i for i in range(list_x.size)} 

        self.dico_y = {list_y[i]: i for i in range(list_y.size)} 

        self.mat = np.zeros((list_x.size, list_y.size, 2))
        
        group = df.groupby([attr1, attr2, 'target']).groups

        for i, j, k in group:
            self.mat[self.dico_x[i], self.dico_y[j], k] = len(group[(i, j, k)]) 
        quant = self.mat.sum(0)
        quant[quant == 0] = 1 #pour eviter des divisions par zero
        
        self.mat = self.mat / quant.reshape((1, list_y.size, 2))
        
    def getProba(self, i, j, k):
        """
        Renvoie la valeur de P(attr1 = i | attr2 = j, target = k).
        Paramètres
        ----------
            i: 
                valeur pour l'attribut attr1 de init.
            j: 
                valeur pour l'attribut attr2 de init.
            k: 
                valeur pour target.
        """
        if i in self.dico_x and j in self.dico_y:
            return self.mat[self.dico_x[i], self.dico_y[j], k]
        return 0.


#####
# 8- Conclusion finale
#####
# Pour conclure l'analyse des méthodes de classifications étudiées ici, on remarque qu'en supposant
# l'indépendance conditionnelle à target mais en ayant target comme unique parent de tous les attributs, les résultats
# en test ne sont pas fiables (comme dit plus haut) mais qu'en théorie, la méthode Tree-augmented Naïve Bayes a les
# meilleurs résultats. En effet, la représentations graphique de MAPTANClassifier se rapproche le plus du point idéal.
# On peut donc en déduire que plus on a de précisions sur la dépendance des attributs, meilleurs sera l'estimation de target.
# Ainsi, les classifieurs bayésiens sont généralement bien adaptés aux données en apprentissage mais ont des résultats
# moins bons en test.
#####
