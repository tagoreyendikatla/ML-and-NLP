import math 
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus={
     'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
}

tokenized_text= [text.lower().split() for text in corpus]

def tf(texts):
    res=[]
    for text in texts:
        word_count=Counter(text)
        total=len(text)
        tf_doc={word:count/total for word, count in word_count.items()}
        res.append(tf_doc)
    return res

def idf(texts):
    n=len(texts)
    idf={}
    all_words=set(word for text in texts for word in text)

    for word in all_words:
        docs_with_word = sum(1 for text in texts if word in text)
        idf[word]=math.log(n/docs_with_word)+1
    return idf

def tfidf(tf, idf):
    tf_idf=[]
    for tf_doc in tf:
        tf_idf_doc={word: tf_value * idf[word] for word, tf_value in tf_doc.items()}
        tf_idf.append(tf_idf_doc)
    return tf_idf

Tf=tf(tokenized_text)
Idf=idf(tokenized_text)
manual=tfidf(Tf, Idf)

Vecotrizer=CountVectorizer()
count_matrix=Vecotrizer.fit_transform(corpus)
count_tf=count_matrix.toarray()

tfidf_vectorizer=TfidfVectorizer()
tfidf_matrix=tfidf_vectorizer.fit_transform(corpus)
sklearn_tfidf= tfidf_matrix.toarray()

print(manual)
print(count_tf)
print(sklearn_tfidf)