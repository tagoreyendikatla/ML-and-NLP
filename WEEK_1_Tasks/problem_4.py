import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def generateReview(sentiment):
    positive_word=["great movie", "highly recommended", "best film of the year", "amazing performace", "loved evrything", "fantastic story", "must watch", "excellent direction","brilliant acting", "interesting plot"]
    negative_word=["terrible movie", "poor acting", "waste of time", "boring plot", "worst film ever", "disappointed", "fell asleep", "bad direction", "awful script","not worth watching"]
    phrase=positive_word if sentiment == "positive" else negative_word
    review=" ".join(np.random.choice(phrase, size=np.random.randint(2,4), replace=True))
    return review.capitalize()+"."

positive_reviews=[generateReview("positive") for i in range(50)]
negative_reviews=[generateReview("negative") for i in range(50)]
data={
    "Review": positive_reviews+negative_reviews,
    "Sentiment": ["positive"]*50 +["negative"]*50
}
df=pd.DataFrame(data)

vectorizer=CountVectorizer(max_features=500, stop_words="english")
x=vectorizer.fit_transform(df["Review"])
y=df["Sentiment"]

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)
mod=MultinomialNB()
mod.fit(x_train, y_train)

predictions=mod.predict(x_test)
accuracy=accuracy_score(y_test, predictions)
print(accuracy)

def predict_review_sentiment(model, vectorizer, review):
    x=vectorizer.transform([review])
    prediction=model.predict(x)
    return prediction
review="This movie was fantastic and well acted"
print(predict_review_sentiment(mod, vectorizer,review))