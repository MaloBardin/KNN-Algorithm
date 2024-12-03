import math
import pandas as pd
import random

from imblearn.over_sampling import SMOTE

#smoteenn et smotetomek : méthodes qui permettent de rééquilibrer les classes avec de l'oversampling et de l'undersampling
from imblearn.combine import SMOTETomek
#over et undersampling combinés avec smote et tomek (oversampling avec smote et cleanup avec les liens tomek)
from imblearn.combine import SMOTEENN
#over et undersampling combinés avec smote et enn (oversampling avec smote et cleanup avec les Edited Nearest Neighbours)

from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.pipeline import Pipeline

def normal_minmax(data): #normalisation minmax (très bon résultats)
  for col in data.columns[1:8]: #normalise uniquement les colonnes C1 à C7 (évite de normaliser l'Id et le label)
    data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
  return data

def normal_iqr(data): #normalisation iqr (nul on le garde pas)
  for col in data.columns[1:8]: #normalise uniquement les colonnes C1 à C7 (évite de normaliser l'Id et le label)
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    data[col] = (data[col] - q1) / iqr
  return data

def normal_zscore(data): #normalisation zscore (résultats corrects mais moins bon que minmax)
  for col in data.columns[1:8]: #normalise uniquement les colonnes C1 à C7 (évite de normaliser l'Id et le label)
    data[col] = (data[col] - data[col].mean()) / data[col].std()
  return data

data = pd.read_csv('train.csv')
data2 = normal_minmax(data)
#data2 = normal_iqr(data)
#data2 = normal_zscore(data)
#data2 = normal_minmax(data2)


X = data2.iloc[:, :-1]  # inclut index et les colonnes de C1...C7
y = data2.iloc[:, -1]  # colonne des labels

#on oversample les données pour équilibrer les classes (SMOTETomek ou SMOTEENN)
#smote = SMOTETomek(sampling_strategy='auto') #ici tout auto au final on préfère personnaliser des paramètres

#SMOTE avec paramètres personnalisés
smotee = SMOTE(k_neighbors=3, n_jobs=4)

#tomek avec parametres personnalisés
tomekk = TomekLinks(sampling_strategy='not majority')

#smote = SMOTETomek(smote=smotee, sampling_strategy='auto')
smote = SMOTETomek(smote=smotee, tomek=tomekk, sampling_strategy='auto')

#smote = SMOTEENN(smote=smotee, sampling_strategy='auto')




#tomek = TomekLinks(sampling_strategy='not majority')
#enn = EditedNearestNeighbours(n_neighbors=3)

#combined_method = Pipeline([('smote',smotee),('tomek', tomek),('enn', enn)]) # smote tomek et enn combinés grâce à pipeline

#X,y = combined_method.fit_resample(X, y)

X, y = smote.fit_resample(X, y)


print("type y :", type(y[0]))

liste_type = []
for i in range(0, 4):
    liste_type.append(i)  # on récupère les types unique

print(liste_type)
NombreCategories = len(liste_type)


class Point():  #class point
    def __init__(self, index, C1, C2, C3, C4, C5, C6, C7, label=None):
        self.index = index
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.C6 = C6
        self.C7 = C7
        self.label = label

    def print(self):
        print("Index : ", self.index, " C1 :", self.C1, " Label:", self.label)


# /////////////////////////////////////# Setup data
dico_point = {}  # dico de liste de point initialisation
for i in range(NombreCategories):
    dico_point[i] = []

for i in range(len(X)):
    for j in range(NombreCategories):
        if y.iloc[i] == liste_type[j]:
            dico_point[j].append(
                Point(X.iloc[i, 0], X.iloc[i, 1], X.iloc[i, 2], X.iloc[i, 3], X.iloc[i, 4], X.iloc[i, 5], X.iloc[i, 6],
                      X.iloc[i, 7], liste_type[j])) #on replit le dico avec chaque point


# for i in range(NombreCategories):
# print("Nouvelle liste")
# for j in range(len(dico_point[i])):
# print(dico_point[i][j].print())


# /////////////////////////////////////////# ALGO KNN


# logique à terminer, on prend 1/3 de chaque liste et pour chaque point on calcule sa distance avec tout les autres. On prend k un nombre max à rentrer en hard et on récuper les k points les plus proches et on regarde
# le type du point étudié (quelle catégorie).
# comment garder l'info de la distance ET du type du point dont la distance a été calculée si on sort la liste -> liste de TUPLES

def distance(point1, point2): #distance euclidienne (la plus performante)
    return math.sqrt((point1.C1 - point2.C1) ** 2 + (point1.C2 - point2.C2) ** 2 + (point1.C3 - point2.C3) ** 2 + (point1.C4 - point2.C4) ** 2 + (point1.C5 - point2.C5) ** 2 + (point1.C6 - point2.C6) ** 2 + (point1.C7 - point2.C7) ** 2)


# print(distance(Liste_point_type1[0],Liste_point_type1[1])) #du test
def maximum(l):
    max = l[0]
    posMax = 0
    for i in range(len(l)):
        if l[i] > max:
            max = l[i]
            posMax = i
    return posMax #max classiuque

def traitement_ponderation(liste): # fonction inutilisée, servait à tester pour des optimisations
    listetempo=[]
    for i in range(len(liste)):
        listetempo.append(((liste[i][0]),liste[i][1]))
    return listetempo

def knn(pointAtest, VarK): #coeur du programme
    liste_distance = []
    for i in range(len(dico_point)):
        for j in range(0, len(dico_point[i])):
            distance_tempo = distance(pointAtest, dico_point[i][j])
            if (distance_tempo != 0): #on evite de prendre le meme point sion cela biaise tout
                ponde_point=1/((distance_tempo)**2) #ponderation par la distance
                liste_distance.append((distance_tempo, dico_point[i][j].label,ponde_point))

    liste_distance.sort()  # on trie par ordre croissant les distances avec une liste de tuple comme ça : (distance,type)
    #liste_distance=traitement_ponderation(liste_distance)#inactif pour le moment
    liste_NbPoint = [0] * NombreCategories  # liste vide de n catégorie
    for i in range(VarK):  # varK c'est le nombre de point à prendre en compte (doit etre impair)
        for j in range(NombreCategories):
            if liste_distance[i][1] == liste_type[j]:
                liste_NbPoint[j] += (liste_distance[i][2])
    return maximum(liste_NbPoint)  # donne l'indice i du type renvoyé


#print("Le knn", knn(dico_point[0][30], 150))


# print(knn(dico_point[0][30],50))


# //////////////////////////////////////# Detection des faux tests

def monTestestfaux(VarK): #trouve le mauvais k
    somme = 0
    for j in range(NombreCategories):
        for i in range(len(dico_point[j])):
            if j != knn(dico_point[j][i], VarK):  # J c'est l'indice de mon dico donc le type finalement
                somme += 1
    return somme


def SommeListe(l):
    somme = 0
    for i in l:
        somme += i
    return somme


# ///////////////////// Trouver le meilleur K

ListeKErreur = []
for i in range(3,6):  # entre 3 et 6 sinon pas pertinent
    tuple_tempo = (i, monTestestfaux(i))
    ListeKErreur.append(tuple_tempo)

ListeKErreur.sort(key=lambda x: x[1])
print(ListeKErreur[0])

Kopti = ListeKErreur[0]
print("K optimisé :", Kopti)
print("Taux réussite : ", 100 - float(ListeKErreur[0][1] / len(X)) * 100)


# crée un csv de train avec un nom unique
def creer_csv_train(dico_point, nombre_points=10):

    nom_fichier = f"résultats train_{random.randint(1, 100000)}.csv" #on crée un nom de fichier unique (pour éviter les doublons)

    # Liste pour stocker les résultats (Id et guess)
    resultats = []

    for categorie in dico_point.values():
        for point in categorie[:nombre_points]:
            guess_label = knn(point, Kopti[0])
            resultats.append({'Id': point.index, 'Label': guess_label})

    df_resultats = pd.DataFrame(resultats) #crée le dataframe pandas

    df_resultats.to_csv(nom_fichier, index=False) #on crée le fichier csv
    print(f"Fichier créé : {nom_fichier}")


# creer_csv_train(dico_point)
datatest2 = pd.read_csv('test.csv')

# on normalise les données de test
datatest = normal_minmax(datatest2)
#datatest = normal_iqr(datatest2)
#datatest = normal_zscore(datatest2)
#datatest = normal_minmax(datatest)

# crée un csv de test avec un nom unique
def creer_csv_predictions(datatest, dico_point, VarK):
    nom_fichier = f"résultats test_{random.randint(1, 100000)}.csv"

    resultats = []  # liste de (Id, Label)

    # prédit le label de chaque point dans datatest
    for i in range(len(datatest)):
        point_test = Point(
            datatest.iloc[i, 0],  # Id
            datatest.iloc[i, 1],  # C1
            datatest.iloc[i, 2],  # C2
            datatest.iloc[i, 3],  # C3
            datatest.iloc[i, 4],  # C4
            datatest.iloc[i, 5],  # C5
            datatest.iloc[i, 6],  # C6
            datatest.iloc[i, 7]  # C7
        )

        guess_label = knn(point_test, Kopti[0])

        resultats.append({'Id': point_test.index, 'Label': guess_label}) #ajoute la prédiction aux résultats

    df_resultats = pd.DataFrame(resultats) #crée le dataframe pandas

    df_resultats.to_csv(nom_fichier, index=False) #crée le fichier csv
    print(f"Fichier de résultats créé : {nom_fichier}")


creer_csv_predictions(datatest, dico_point, Kopti[0])