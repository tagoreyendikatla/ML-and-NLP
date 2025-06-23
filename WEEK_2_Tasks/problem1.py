import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

data=pd.read_csv('spam.csv', encoding='latin-1')
data=data[['v1','v2']]
data.columns=['label','message']

stop_words=set(stopwords.words('english'))
preprocessed_message=[]
for message in data['message']:
    message=message.lower()
    message=re.sub(r'[^a-zA-Z\s]','',message)
    tokens=word_tokenize(message)
    tokens=[word for word in tokens if word not in stop_words]
    preprocessed_message.append(tokens)

w2v_model=KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

x=[]
for message in preprocessed_message:
    word_vectors=[]
    for word in message:
        if word in w2v_model:
            word_vectors.append(w2v_model[word])
    if len(word_vectors)>0:
        x.append(np.mean(word_vectors, axis=0))
    else:
        x.append(np.zeros(300))

x=np.array(x)
y=data['label'].map({'ham':0, 'spam':1}).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(x_train, y_train)

predictions= classifier.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

def predict_message_class(model, w2v_model, message):
    message=message.lower()
    message=re.sub(r'[^a-zA-Z\s]','',message)
    tokens=word_tokenize(message)
    tokens=[word for word in tokens if word not in stop_words]

    word_vectors=[]
    for word in tokens:
        if word in w2v_model:
            word_vectors.append(w2v_model[word])
    if len(word_vectors)>0:
        message_vector=np.mean(word_vectors, axis=0)
    else:
        message_vector=np.zeros(300)

    message_vector=message_vector.reshape(1,-1)
    prediction=model.predict(message_vector)

    return 'spam' if prediction[0]==1 else 'ham'