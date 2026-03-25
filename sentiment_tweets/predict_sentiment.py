import re
import joblib

MODEL_PATH = "sentiment_model.pkl"


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-záéíóúñü0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    model = joblib.load(MODEL_PATH)

    tweet = input("Escribe un tweet: ")
    clean_tweet = clean_text(tweet)

    pred = model.predict([clean_tweet])[0]
    probs = model.predict_proba([clean_tweet])[0]

    classes = model.classes_
    print(f"\nTweet: {tweet}")
    print(f"Sentimiento predicho: {pred}")
    print("Probabilidades:")
    for cls, prob in zip(classes, probs):
        print(f"  {cls}: {prob:.4f}")


if __name__ == "__main__":
    main()