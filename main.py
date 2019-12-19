from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
import sys

with open('./imdb_x_clean.txt', encoding='utf8') as f:
    reviews = f.readlines()
with open('./dataset/imdb_trainY.txt', encoding='utf8') as f:
    y = f.readlines()
with open('./imdb_test_clean.txt', encoding='utf8') as f:
    test_rev = f.readlines()


Y = [int(i) for i in y]
cv = CountVectorizer()
X = cv.fit_transform(reviews).toarray()
X_test = cv.transform(test_rev).toarray()
print(X.shape)
mnb = MultinomialNB()
mnb.fit(X, Y)
predictions = mnb.predict(X_test)
predictions = np.array(predictions)
dfp = pd.DataFrame(predictions)
dfp.to_csv('predictions.csv')


