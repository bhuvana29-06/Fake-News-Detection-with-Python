import joblib

model = joblib.load('../model/fake_news_model.pkl')
vectorizer = joblib.load('../model/vectorizer.pkl')

def predict_news(text):
    tfidf = vectorizer.transform([text])
    return model.predict(tfidf)[0]

if __name__ == "__main__":
    sample = input("Enter news text:\n")
    result = predict_news(sample)
    print(f"\n📰 Prediction: {result}")
