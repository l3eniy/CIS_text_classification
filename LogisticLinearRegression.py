import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import nltk
from nltk.tokenize import TreebankWordTokenizer
from scipy.sparse import csr_matrix

df = pd.read_csv("BigDataTemplate4.txt", names=['CIS-Req', 'label'], sep='\t')
# df.info()

requirement_description = df['CIS-Req'].values
funktionales_requirement = df['label'].values

# test_size 0.4 zeigt beste Ergebnisse
requirement_description_train, requirement_description_test, funktional_train, funktional_test = train_test_split(requirement_description, funktionales_requirement,test_size=0.4, random_state=45)

### Preprocessing / Daten vorbereiten
vectorizer = CountVectorizer()
vectorizer.fit(requirement_description_train)


requirement_description_train = vectorizer.transform(requirement_description_train)
requirement_description_test = vectorizer.transform(requirement_description_test)


### Modell auswählen und anwenden
# classifier = LogisticRegression()
# classifier = RandomForestClassifier()
# classifier = GradientBoostingClassifier()
classifier = SVC()
classifier.fit(requirement_description_train, funktional_train)     # Hier wird das ausgewählte Modell mit den trainingsdaten trainiert

### Präzision es Modells auswerten
funktional_prediction = classifier.predict(requirement_description_test)
print(classification_report(funktional_test,funktional_prediction))
print(confusion_matrix(funktional_test,funktional_prediction))




score1 = classifier.score(requirement_description_train, funktional_train)
score2 = classifier.score(requirement_description_test, funktional_test)

# print("Genauigkeit der Trainingsdaten: " + str(score1))
# print("Genauigkeit der Testdaten: "+ str(score2))

