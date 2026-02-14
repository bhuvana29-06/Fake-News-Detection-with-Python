from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

import clean_data

def train():
    df = clean_data.load_data('../data/news.csv')

    x_train, x_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=7
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = vectorizer.fit_transform(x_train)
    tfidf_test = vectorizer.transform(x_test)

    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(tfidf_train, y_train)

    y_pred = model.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)

    print(f'➡️ Model Accuracy: {round(score * 100, 2)}%')

    # Confusion Matrix
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred, labels=['FAKE','REAL']))

    joblib.dump(model, '../model/fake_news_model.pkl')
    joblib.dump(vectorizer, '../model/vectorizer.pkl')

if __name__ == "__main__":
    train()
