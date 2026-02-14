import joblib
from sklearn.metrics import classification_report
import clean_data

def evaluate():
    model = joblib.load('../model/fake_news_model.pkl')
    vectorizer = joblib.load('../model/vectorizer.pkl')

    df = clean_data.load_data('../data/news.csv')
    X = df['text']
    y = df['label']

    X_tfidf = vectorizer.transform(X)
    y_pred = model.predict(X_tfidf)

    print(classification_report(y, y_pred))

if __name__ == "__main__":
    evaluate()
