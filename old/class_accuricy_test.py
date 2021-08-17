import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.tree import DecisionTreeClassifier

list_X = ["eiche", "erle", "Pudel","Banane", "Birke", "Fichte", "erdbeere", "Schäferhund", "Apfel", "Traube"]
list_Y = ["baum", "baum", "hund", "frucht", "baum", "baum", "frucht", "hund", "frucht", "frucht"]

# global verfügbare Objekte
VECTORIZER = CountVectorizer()  # Muss global verfügbar sein
# CLASSIFIER = SVC()
CLASSIFIER = DecisionTreeClassifier()
# CLASSIFIER = LogisticRegression()
# CLASSIFIER = RandomForestClassifier()
# CLASSIFIER = GradientBoostingClassifier()


def prepare_trainingsdata(cis_list, funktional_list):
    """
    Die Funktion macht das preprocessing mit der Hilfe von X und Y Liste. Zurückgegeben werden Daten die der Classifier versteht
    :param cis_list: Die X-Werte --> example: ["Ensure mounting of df is disabled", "No ssh root must be used", "Dont use lorem ipsum", ...]
    :param funktional_list: Die zugehörigen Y-Werte sind die Label: example: [0, 1, 0, 1, 2, 2]
    :return: cis_requirement_vectorized --> scipy.sparse.csr.csr_matrix
    :return: funktional_list            --> eigentlich unnötig
    :return: vectorizer                 --> der verwendete CountVectorizer. --> WICHTIG: Man muss immer den selben Count Vectorizer verwenden
    """
    VECTORIZER.fit(cis_list)
    cis_requirement_vectorized = VECTORIZER.transform(cis_list)
    return cis_requirement_vectorized, funktional_list


def predict_funktionale_Anforderung(neuer_y_string):
    """
    Funktion bestimmt das zugehörige Label. Sie verwendet den globalen Vectorizer und den übergebenen classifier.
    :param neuer_y_string: Ein String zu dem das Label gefunden werden soll
    :param classifier: Der verwendete Classifier
    :return: Gibt den Wert des Labels zurück
    """
    X_test = VECTORIZER.transform([neuer_y_string])
    predicted_label = CLASSIFIER.predict(X_test)
    print(neuer_y_string + "\t\t-->\t\t" + str(predicted_label[0]))
    return predicted_label[0]


def print_mapping(X, Y, filter):
    '''
    Gibt das Mapping von X und Y in Konsole aus. Mit filter auf "no" wird alles ausgegeben
    :param X: CIS_Requirement
    :param Y: funktionale
    :param filter: "no" --> alles, int --> filtert Y
    :return: None
    '''
    for i in range(0, len(X)):
        if filter == "no":
            print(X[i] + "\t\t-->\t\t" + str(Y[i]))
            continue
        if Y[i] == filter:
            print(X[i] + "\t\t-->\t\t" + str(Y[i]))


X, Y = prepare_trainingsdata(list_X, list_Y)
CLASSIFIER.fit(X, Y)  # Hier wird das ausgewählte Modell mit den trainingsdaten trainiert

test_X = "bananenhund"


print(CLASSIFIER.predict_proba(VECTORIZER.transform([test_X])))

predict_funktionale_Anforderung(test_X)




