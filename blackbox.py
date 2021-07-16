import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump, load
import pickle



df = pd.read_csv("BigDataTemplate4.txt", names=['CIS-Req', 'label'], sep='\t')
cis_requirements = df['CIS-Req'].values
funktionale_requirements = df['label'].values

list_X = ["Heute ist das Wetter nicht gut", "Heute ist das Wetter gut", "Heute das Wetter ist nicht gut",
           "Das Wetter ist heute gut", "Kopfschmerzen sind schlimm!", "Heute habe ich Kopfschmerzen wegen dem Wetter"]
list_Y = [0, 1, 0, 1, "String geht auch als Label", "String geht auch als Label"] # 0 --> schlechtes Wetter, 1 --> gutes Wetter, 3 --> Kopfschmerzen




# global verfügbare Objekte
VECTORIZER = CountVectorizer()  # Muss global verfügbar sein
CLASSIFIER = SVC()
# classifier = LogisticRegression()
# classifier = RandomForestClassifier()
# classifier = GradientBoostingClassifier()




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

def print_mapping(X,Y, filter):
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



X, Y = prepare_trainingsdata(cis_requirements, funktionale_requirements)
CLASSIFIER.fit(X, Y)  # Hier wird das ausgewählte Modell mit den trainingsdaten trainiert

print_mapping(cis_requirements, funktionale_requirements, 1)

test_X = "Ensure foobar services are not enabled"
predict_funktionale_Anforderung(test_X)


# save Classifier persistent
with open("test.pickle", "wb") as file:
    pickle.dump(CLASSIFIER, file, protocol=pickle.HIGHEST_PROTOCOL)