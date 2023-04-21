from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask import Flask, request, jsonify
import os
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_sentiment(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )
    inputs = inputs.to(device)
    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()[0]
    predicted_sentiment = int(logits.argmax(axis=-1))
    probabilities = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
    negative_percentage = round(probabilities[0]*100, 4)
    positive_percentage = round(probabilities[2]*100, 4)
    neutral_percentage = round(probabilities[1]*100, 4)
    return negative_percentage, neutral_percentage, positive_percentage

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World from cletus!'


@app.route('/predict_sentiment', methods=['POST'])
def predict():
    text = request.json['text']
    negative_percentage, neutral_percentage, positive_percentage = predict_sentiment(text)
    response = {
        'negative_percentage': negative_percentage,
        'neutral_percentage': neutral_percentage,
        'positive_percentage': positive_percentage
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
