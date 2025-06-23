import pandas as pd
import numpy as np
import re
from contractions import fix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk


df=pd.read_csv('Tweets.csv')
df=df[['airline_sentiment','text']]
df.columns=['sentiment','tweet']

lemmatizer=WordNetLemmatizer()
stop_words=set(stopwords.words('english'))

preprocessed_tweet=[]
for tweet in df['tweet']:
    text=str(tweet).lower()
    text=re.sub(r'http\S+|www\S+|@\w+|#\w+','',text)
    text=fix(text)
    text=re.sub(r'[^a-z\s]','',text)
    tokens=[lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words]
    preprocessed_tweet.append(tokens)

w2v_model=KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)

x=[]
for tweet in preprocessed_tweet:
    vectors=[w2v_model[word] for word in tweet if word in w2v_model]
    x.append(np.mean(vectors, axis=0) if vectors else np.zeros(300))
x=np.array(x)

y=df['sentiment'].map({'negative':0, 'neutral':1,'positive':2}).values

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)

classifier=LogisticRegression(multi_class='multinomial', max_iter=1000)
classifier.fit(x_train, y_train)

predictions=classifier.predict(x_test)
