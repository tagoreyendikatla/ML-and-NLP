import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

def generatefeedback(label):
    positive_feedback=["great product", "highly recommended", "excellent quality", "works perfectly", "very satisfied", "good value", "love it", "exceeds expectation", "amazing performance", "reliable and durable", "perfect fit", "fast delivery"]
    negative_feedback=["poor quality", "stopped working", "not worth it", "disappointed", "waste of money", "defective", "not as described", "fake product", "returning this", "broken on arrival", "doesn't work", "terrible"]

    feedbacks=positive_feedback if label == 'good' else negative_feedback
    feedback=" ".join(np.random.choice(feedbacks, size=np.random.randint(2,4), replace=True))
    return feedback.capitalize()+"."

good_feedback= [generatefeedback("good") for i in range(50)]
bad_feedback=[generatefeedback("bad") for i in range(50)]

data={
    "text": good_feedback+bad_feedback,
    "label": ["good"]*50 + ["bad"]*50
}
df=pd.DataFrame(data)

pipe=Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=300, stop_words='english', lowercase=True)),
    ('classifier', LogisticRegression())
])

x_train, x_test, y_train, y_test= train_test_split(df['text'], df['label'], test_size=0.25, random_state=42)

pipe.fit(x_train, y_train)

predictions=pipe.predict(x_test)

precision=precision_score(y_test, predictions, pos_label='good')
recall= recall_score(y_test, predictions, pos_label='good')
f1=f1_score(y_test, predictions, pos_label='good')

print(precision)
print(recall)
print(f1)

def text_preprocess_vectorize(texts, pipe):
    return pipe.named_steps['vectorizer'].transform(texts)
text1=["this product is amazing", "Terrible experience"]
print(text_preprocess_vectorize(text1, pipe))