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
from openpyxl import Workbook
import xlsxwriter


### PLOT PARAMETERS
figure(figsize=(15, 6), dpi=120)
PLOTTITEL = "Wahrscheinlichkeitsverteilung der DTAG Funktionalen Anforderungen <--> CIS-Anforderung:\n"
YLABEL = "Wahrscheinlichkeit"
XLABEL = "DTAG Funktionale Anforderung"

### READ DATA from CSV
df = pd.read_csv("BigDataTemplate4.txt", names=['CIS-Req', 'label'], sep='\t')
cis_requirements = df['CIS-Req'].values
funktionale_requirements = df['label'].values

### READ TEST-DATA from CSV
reading_testdata = pd.read_csv("CIS_WindowsReq.txt", names=['CIS-Req'], sep='\t', encoding="UTF-8", engine='python')  # ensure rsyslog is used for remote log  vs.  ensure rsyslog is used for remote logging
test_X = reading_testdata['CIS-Req'].values

### READ TEST-DATA from XML

### READ FUNCTIONAL REQUIREMENTS FROM CSV
read_functional = pd.read_csv("Funktionale_Anforderungen_OS_en.csv", names=['FR'],  sep='\t', engine='python')
functional_text = read_functional['FR'].values


# Wie viele CIS_Reqs sind den Funktionalen jeweils zugeordnet? :
# print("label\tcount")
# print(df['label'].value_counts())


### MEINE TESTDATEN
list_X = ["Heute ist das Wetter nicht gut", "Heute ist das Wetter gut", "Heute das Wetter ist nicht gut",
          "Das Wetter ist heute gut", "Kopfschmerzen sind schlimm!", "Heute habe ich Kopfschmerzen wegen dem Wetter"]
list_Y = ["0", "1", "0", "1", "Kopfschmerzen",
          "Kopfschmerzen"]  # 0 --> schlechtes Wetter, 1 --> gutes Wetter, 3 --> Kopfschmerzen

### global verfügbare Objekte
VECTORIZER = CountVectorizer(max_df=0.25,
                             ngram_range=(1, 1))  # Muss global verfügbar sein # FEATURE ENGINEERING DURCH PARAMETER
# CLASSIFIER = SVC()
# CLASSIFIER = LogisticRegression()
CLASSIFIER = RandomForestClassifier()


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
    # print("[+] " + neuer_y_string + "\t\t-->\t\t" + str(predicted_label[0]))
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


def save_persistent_model(pickle_file_name):
    with open(pickle_file_name, "wb") as file:
        pickle.dump(CLASSIFIER, file, protocol=pickle.HIGHEST_PROTOCOL)


def use_persistent_model(pickle_file_name):
    with open(pickle_file_name, 'rb') as handle:
        classifier_from_pickle = pickle.load(handle)
    return classifier_from_pickle


def get_prediction_probability_for_sample(sample):
    X_test = VECTORIZER.transform([sample])
    prediction_probability = CLASSIFIER.predict_proba(X_test)
    return prediction_probability[0]


def get_probability_plot(values, testdata_x):
    names = ["N/A"]
    for i in range(1, 35):
        names.append("f" + str(i))

    plt.bar(names, values)
    plt.title(PLOTTITEL + testdata_x)
    plt.ylabel(YLABEL)
    plt.xlabel(XLABEL)
    plt.show()


### TESTEN DES MODELLS
def test_excel_windows(test_X, functional_text):
    wb = xlsxwriter.Workbook("Excel1.xlsx")
    ws = wb.add_worksheet()
    MagentaBgColorCellFormat = wb.add_format()
    MagentaBgColorCellFormat.set_bg_color('ea0a8e')
    LightBlueBgColorCellFormat = wb.add_format()
    LightBlueBgColorCellFormat.set_bg_color('add8e6')
    row = 0
    col = 1
    for e in functional_text:
        ws.write(row, col, e, MagentaBgColorCellFormat)
        col += 1
    row += 1

    for cis_requirements in range(len(test_X)):
        prediction_probabilities = get_prediction_probability_for_sample(test_X[cis_requirements])
        #get_probability_plot(prediction_probabilities, test_X[cis_requirements])
        predict_funktionale_Anforderung(test_X[cis_requirements])

        ###Definition of the three Highest Possibility Values
        Wkt_zu_Liste = list(prediction_probabilities)
        Wkt_zu_Liste.sort()
        Q = Wkt_zu_Liste[-3:]
        for e in test_X:
            col = 0
            ws.write(row, col, test_X[cis_requirements], LightBlueBgColorCellFormat)
            col += 1
        col = 1

        for i in prediction_probabilities:
            if i == Q[0]:
                redBgColorCellFormat = wb.add_format()
                redBgColorCellFormat.set_bg_color('red')
                ws.write(row, col, i, redBgColorCellFormat)
                col += 1
            elif i == Q[1]:
                redBgColorCellFormat = wb.add_format()
                redBgColorCellFormat.set_bg_color('red')
                ws.write(row, col, i, redBgColorCellFormat)
                col += 1
            elif i == Q[2]:
                redBgColorCellFormat = wb.add_format()
                redBgColorCellFormat.set_bg_color('red')
                ws.write(row, col, i, redBgColorCellFormat)
                col += 1
            else:
                ws.write(row, col, 'N/A')
                col += 1
        row += 1
    wb.close()




### TRAINING DES MODELLS
# CLASSIFIER = use_persistent_model("test1.pickle")
X, Y = prepare_trainingsdata(cis_requirements, funktionale_requirements)
CLASSIFIER.fit(X, Y)  # Hier wird das ausgewählte Modell mit den trainingsdaten trainiert

#print_mapping(cis_requirements, funktionale_requirements, 34)

###TESTEN DES MODELLS
test_excel_windows(test_X, functional_text)

print("[+] Folgende Stopwords wurden entfernt:  " + str(VECTORIZER.stop_words_))

# save Classifier persistent as pickle file
# save_persistent_model("test1.pickle")
