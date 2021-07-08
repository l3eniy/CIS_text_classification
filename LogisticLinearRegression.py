import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import nltk
from nltk.tokenize import TreebankWordTokenizer
from scipy.sparse import csr_matrix

df = pd.read_csv("BigDataTemplate4.txt", names=['CIS-Req', 'label'], sep='\t')

requirement_description = df['CIS-Req'].values
funktionales_requirement = df['label'].values

scores = []
for i in range(1,100):
    i = i/100

    requirement_description_train, requirement_description_test, funktional_train, funktional_test = train_test_split(requirement_description, funktionales_requirement,test_size=i, random_state=45)

    vectorizer = CountVectorizer()
    vectorizer.fit(requirement_description_train)

    requirement_description_train = vectorizer.transform(requirement_description_train)
    requirement_description_test = vectorizer.transform(requirement_description_test)

    classifier = LogisticRegression()
    # classifier = RandomForestClassifier()
    # classifier = GradientBoostingClassifier()
    classifier.fit(requirement_description_train, funktional_train)
    score1 = classifier.score(requirement_description_train, funktional_train)
    score2 = classifier.score(requirement_description_test, funktional_test)

    print("################################## Anteil der Testdaten = " + str(i))
    print("Genauigkeit der Trainingsdaten: " + str(score1))
    print("Genauigkeit der Testdaten: "+ str(score2))
    scores.append(score1+score2)
print("Max Score: " + str(max(scores)))
