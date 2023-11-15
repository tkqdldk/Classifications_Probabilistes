
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
# Nous préférons MAP2DClassifier parce que nous cherchons à faire diminuer le nombre de faux positifs
# et aussi parce que
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
        None
    """
    taille = 8
    res = 0
    if liste is None:
        liste = list(df.columns)
    for attr in liste:
        taille *= np.unique(df[attr]).size
    print("{} variable(s) : {} octets".format(len(liste), taille))
    return taille # pas sûre a tester


def nbParamsIndep(df):
    """
    Calcule le nombre de paramètres nécessaires pour représenter les données du dataframe.
    
    Paramètres
    ----------
        df : pandas.DataFrame
         Le dataframe contenant les données
    
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
# Question 3.3.a
#####
# Preuve:
# ------
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
# Question 3.3.b
#####
# Si les 3 variables A,B et C ont 5 valeurs de taille 8 octets.
# Avec l'indépendance conditionnelles, il faut (5^3)*8=1000 octets pour représenter ces valeurs en mémoire.
# Tandis que sans l'indépendance en mémoire, il faut (5+2*(5^2))*8=440 octets pour les représenter.
#####

#####
# Question 4.1
#####
# Voici notre proposition de code pour dessiner les graphes pour 5 variables complètement indépendantes:
# utils.drawGraphHorizontal("A;B;C;D;E")
# Voici notre proposition pour dessiner les graphes pour 5 variables sans indépendance:
# utils.drawGraphHorizontal("A->B;B->C;C->D;D->E;E->A")
#####

#####
# Question 4.2
#####
#
#
#####

def drawNaiveBayes(df, root):
    """
    Construit un graphe orienté représentant naïve Bayes.
    
    Paramètres
    ----------
        df: pandas.dataframe
            Dataframe de reference
        root: str
            Le nom de la colonne du Dataframe utilisée comme racine.
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


def nbParamsNaiveBayes(df, root, attrs = None):
    """
    Affiche la taille memoire pour la representation des tables de probabilites
    Parametres
    ----------
        df : pandas.dataframe
            DataFrame de reference
        root : str
            Nom de la c
        attrs : Dict
    Resultats
    ---------
        int

    """
    taille = np.unique(df[root]).siza
    size = taille * 8

    if attrs == None:
        attrs = list.df[colonne]

    for attr in attrs:
        if attr == root:
            continue
        tmp = (taille * np.unique(df[attr]).size) * 8
        size += tmp

    print("{} variable(s) : {} octets".format(len(attrs), size))
    return size #encore une fois pas sûre

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
            dataframe. Doit contenir une colonne appelée "target".
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
            attrs: dictionnaire nom-valeur des attributs

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
            attrs: le dictionnaire nom-valeur des attributs

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
            df: dataframe. Doit contenir une colonne appelée "target".
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
            attrs: le dictionnaire nom-valeur des attributs
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
        Calcule la probabilité à posteriori par naïve Bayes : P(target | attr1, ..., attrk).
        Paramètres
        ----------
            attrs: le dictionnaire nom-valeur des attributs
        Résultats
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
        df: dataframe. Doit contenir une colonne appelée "target".
        attr: le nom d'une colonne du dataframe df.
        x: seuil de confiance.

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
            df: dataframe. Doit contenir une colonne appelée "target".
            x: seuil de confiance pour le test d'indépendance
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
            attrs: le dictionnaire nom-valeur des attributs
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
            attrs: le dictionnaire nom-valeur des attributs
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
            df: dataframe. Doit contenir une colonne appelée "target".
            x: seuil de confiance pour le test d'indépendance.
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
            attrs: le dictionnaire nom-valeur des attributs
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
            attrs: le dictionnaire nom-valeur des attributs

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
# 
# 
# 
#####


def mapClassifiers(dic, df):
    """
    Représente graphiquement les classifieurs dans l'espace (précision, rappel).
    Paramètres
        ----------
        dic: Dictionnaire {nom: instance de classifier}.
        df: DataFrame. Doit contenir une colonne appelée "target".
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
# Question 6.3
#####
# Conclusion:

#####




def MutualInformation(df, x, y):

    """
    Calcule l'information mutuelle entre les colonnes x et y du dataframe.
    Paramètres
    ----------
        df: Dataframe contenant les données. 
        x: nom d'une colonne du dataframe.
        y: nom d'une colonne du dataframe.
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
        df: Dataframe contenant les données. 
        x: nom d'une colonne du dataframe.
        y: nom d'une colonne du dataframe.
        z: nom d'une colonne du dataframe.
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
