#code for logistic regression
#nur ein feature berücksichtigt über .txt datei eingelesen, accuracy liegt bei 40%
#Keras und Tensorflow  implementiert
#Score Ergebnisse mit Keras:training accuracy:0.0837
#training accuracy:0.1324

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras import layers
from keras.backend import clear_session


filepath_dict = {'CIS_labeled': 'BigDataTemplate4.txt'
}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names = ['CIS', 'label'], sep= '\t')
    df_list.append(df)

df = pd.concat(df_list)
print(df.iloc[0])


vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(df['CIS'])
print(vectorizer.vocabulary_)

print(vectorizer.transform(df['CIS']).toarray)
requirement_description = df['CIS'].values
y = df['label'].values

requirement_description_train, requirement_description_test, y_train, y_test = train_test_split(requirement_description, y, test_size = 0.25, random_state = 1000)

vectorizer = CountVectorizer()
vectorizer.fit(requirement_description_train)

X_train = vectorizer.transform(requirement_description_train)
X_test = vectorizer.transform(requirement_description_test)

X_train

#classifier = LogisticRegression()
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)
score1 = classifier.score(X_train , y_train)
score2 = classifier.score(X_test, y_test)

print("Accuracy: Train", score1)
print("Accuracy: Test", score2)


input_dim = X_train.shape[1]
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=100,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10
                    )

clear_session()

loss, accuracy = model.evaluate(X_train,y_train, verbose=False)
print("training accuracy:{:.4f}".format(accuracy))

loss1, accuracy2 = model.evaluate(X_test, y_test, verbose=False)
print("training accuracy:{:.4f}".format(accuracy2))




