import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#from keras.models import Sequential
#from keras import layers
#from keras.backend import clear_session
#import nltk
from nltk.tokenize import TreebankWordTokenizer
from scipy.sparse import csr_matrix


filepath_dict = {'CIS_labeled': 'BigDataTemplate4.txt'
                 }

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names = ['CIS-Req', 'label'], sep= '\t')
    df_list.append(df)

df = pd.concat(df_list)
print(df.iloc[0])

#tokenizer = TreebankWordTokenizer(df)
#print (tokenizer)


vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(df['CIS-Req'])
print(vectorizer.vocabulary_)
print(vectorizer.transform(df['CIS-Req']).toarray)

requirement_description = df['CIS-Req'].values
y = df['label'].values

requirement_description_train, requirement_description_test, y_train, y_test = train_test_split(requirement_description, y, test_size = 0.25, random_state = 1000)

vectorizer = CountVectorizer()
vectorizer.fit(requirement_description_train)

X_train = vectorizer.transform(requirement_description_train)
X_test = vectorizer.transform(requirement_description_test)

X_train

#classifier = LogisticRegression()
#classifier = RandomForestClassifier()
classifier = GradientBoostingClassifier()
classifier.fit(X_train, y_train)
score1 = classifier.score(X_train, y_train)
score2 = classifier.score(X_test, y_test)

print(type(X_train))
print(type(y_train))
#print(y_train)
print(csr_matrix(X_train).toarray())

print("Accuracy: Train", score1)
print("Accuracy: Test", score2)





