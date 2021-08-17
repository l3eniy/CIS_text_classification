#!/usr/bin/python3

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

### PLOT PARAMETERS
figure(figsize=(15, 6), dpi=120)
PLOTTITEL = "Wahrscheinlichkeitsverteilung der DTAG Funktionalen Anforderungen <--> CIS-Anforderung:\n"
YLABEL = "Wahrscheinlichkeit"
XLABEL = "DTAG Funktionale Anforderung"

# Wie viele CIS_Reqs sind den Funktionalen jeweils zugeordnet? :
# print("label\tcount")
# print(df['label'].value_counts())


### MEINE TESTDATEN
list_X = ["Heute ist das Wetter nicht gut", "Heute ist das Wetter gut", "Heute das Wetter ist nicht gut",
          "Das Wetter ist heute gut", "Kopfschmerzen sind schlimm!", "Heute habe ich Kopfschmerzen wegen dem Wetter"]
list_Y = ["0", "1", "0", "1", "Kopfschmerzen",
          "Kopfschmerzen"]  # 0 --> schlechtes Wetter, 1 --> gutes Wetter, 3 --> Kopfschmerzen

class blackbox:
    cis_requirements = None
    funktionale_requirements = None
    ### global verfügbare Objekte
    VECTORIZER = CountVectorizer(max_df=0.25, ngram_range=(1,1))  # Muss global verfügbar sein # FEATURE ENGINEERING DURCH PARAMETER
    # CLASSIFIER = SVC()
    # CLASSIFIER = LogisticRegression()
    CLASSIFIER = RandomForestClassifier()
    # CLASSIFIER = GradientBoostingClassifier()

    def prepare_trainingsdata(self,cis_list, funktional_list):
        """
        Die Funktion macht das preprocessing mit der Hilfe von X und Y Liste. Zurückgegeben werden Daten die der Classifier versteht
        :param cis_list: Die X-Werte --> example: ["Ensure mounting of df is disabled", "No ssh root must be used", "Dont use lorem ipsum", ...]
        :param funktional_list: Die zugehörigen Y-Werte sind die Label: example: [0, 1, 0, 1, 2, 2]
        :return: cis_requirement_vectorized --> scipy.sparse.csr.csr_matrix
        :return: funktional_list            --> eigentlich unnötig
        :return: vectorizer                 --> der verwendete CountVectorizer. --> WICHTIG: Man muss immer den selben Count Vectorizer verwenden
        """
        self.VECTORIZER.fit(cis_list)
        self.cis_requirement_vectorized = self.VECTORIZER.transform(cis_list)
        return self.cis_requirement_vectorized, funktional_list


    def predict_funktionale_Anforderung(self,neuer_y_string):
        """
        Funktion bestimmt das zugehörige Label. Sie verwendet den globalen Vectorizer und den übergebenen classifier.
        :param neuer_y_string: Ein String zu dem das Label gefunden werden soll
        :param classifier: Der verwendete Classifier
        :return: Gibt den Wert des Labels zurück
        """
        X_test = self.VECTORIZER.transform([neuer_y_string])
        predicted_label = self.CLASSIFIER.predict(X_test)
        print("[+] " + neuer_y_string + "\t\t-->\t\t" + str(predicted_label[0]))
        return predicted_label[0]


    def print_mapping(self,X, Y, filter):
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


    def save_persistent_model(self,pickle_file_name):
        with open(pickle_file_name, "wb") as file:
            pickle.dump(self.CLASSIFIER, file, protocol=pickle.HIGHEST_PROTOCOL)


    def use_persistent_model(self,pickle_file_name):
        with open(pickle_file_name, 'rb') as handle:
            classifier_from_pickle = pickle.load(handle)
        return classifier_from_pickle


    def get_prediction_probability_for_sample(self,sample):
        X_test = self.VECTORIZER.transform([sample])
        prediction_probability = self.CLASSIFIER.predict_proba(X_test)
        return prediction_probability[0]

    def get_probability_map(self,values):
        names = ["N/A"]
        data = {}
        for i in range(1, 35):
            names.append("f" + str(i))
        for i in range(0, 35):
            data[names[i]] = values[i]
        return data 

    def get_probability_plot(self,values, testdata_x):
        names = ["N/A"]
        for i in range(1, 35):
            names.append("f" + str(i))

        plt.bar(names, values)
        plt.title(PLOTTITEL + testdata_x)
        plt.ylabel(YLABEL)
        plt.xlabel(XLABEL)
        plt.show()
    
    def __init__(self):
        ### TRAINING DES MODELLS
        # CLASSIFIER = use_persistent_model("test1.pickle")
        ### READ DATA from CSV
        df = pd.read_csv("BigDataTemplate4.txt", names=['CIS-Req', 'label'], sep='\t')
        self.cis_requirements = df['CIS-Req'].values
        self.funktionale_requirements = df['label'].values

        X, Y = self.prepare_trainingsdata(self.cis_requirements, self.funktionale_requirements)
        self.CLASSIFIER.fit(X, Y)  # Hier wird das ausgewählte Modell mit den trainingsdaten trainiert
    
    def dump_mapping(self):
        self.print_mapping(self.cis_requirements,self.funktionale_requirements,34)


# print_mapping(cis_requirements, funktionale_requirements, 33)


if __name__ == '__main__':
    ### TESTEN DES MODELLS
    bb = blackbox()
    test_X = "ensure log is empty" # ensure rsyslog is used for remote log  vs.  ensure rsyslog is used for remote logging

    prediction_probabilities = bb.get_prediction_probability_for_sample(test_X)
    print(prediction_probabilities)
    print(bb.get_probability_map(prediction_probabilities))
    bb.get_probability_plot(prediction_probabilities, test_X)

    bb.predict_funktionale_Anforderung(test_X)
    print("[+] Folgende Stopwords wurden entfernt:  " + str(bb.VECTORIZER.stop_words_))

    # save Classifier persistent as pickle file
    # save_persistent_model("test1.pickle")
