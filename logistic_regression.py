import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

#Read in the csv files into pandas data frames
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#These values are the text attributes of the tuples all lowercase lettering
train['lower_case_text'] = train.text.map(lambda x: x.lower() if isinstance(x,str) else x)
test['lower_case_text'] = test.text.map(lambda x: x.lower() if isinstance(x,str) else x)

#Tokenize the tweets based on the tweet tokenizer provided in the nltk library
tokenizer = TweetTokenizer().tokenize
vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', ngram_range=(1, 3), max_df=0.3)
vectorizer.fit(train['lower_case_text'].values)
train_tfidf = vectorizer.transform(train['lower_case_text'].values)
test_tfidf = vectorizer.transform(test['lower_case_text'].values)

targets = train.target.values.copy()
logreg = LogisticRegression(n_jobs=-1,C= 10, class_weight= 'balanced', penalty= 'l2', random_state= 42)

logreg.fit(train_tfidf, targets)
preds = logreg.predict_proba(test_tfidf)
ids = np.asarray(test.id.values)
values = np.argmax(preds,axis = 1)
DF = pd.DataFrame({'id': ids, 'target': values}, columns = ['id', 'target'])
DF.to_csv('results2.csv', index = False)