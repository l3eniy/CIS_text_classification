import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

print("Accuracy", score)



