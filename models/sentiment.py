import csv
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


class FinBertSentimentAnalyzer:
    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
            self.model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
            self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            logging.info(
                "FinBERT sentiment model initialized",
                extra={"pipeline_step": "model_load"},
            )
        except Exception as e:
            logging.error(
                f"Failed to initialize FinBERTSentimentAnalyzer: {e}",
                extra={"error_category": "model"},
            )
            raise

    def predict_with_confidence(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length",
        )
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
        sentiment_score = probs[2] - probs[0]  # positive - negative
        confidence = float(np.max(probs))
        label = self.label_map[int(np.argmax(probs))]
        return label, sentiment_score, confidence

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length",
        )
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
        sentiment_score = probs[2] - probs[0]  # positive - negative
        return float(sentiment_score)


def evaluate_sentiment_model(model, validation_path="models/finance_sentiment_validation.csv", confidence_threshold=0.6):
    y_true, y_pred, y_conf = [], [], []
    with open(validation_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            text, label = row[0], row[1]
            pred_label, _, conf = model.predict_with_confidence(text)
            if conf < confidence_threshold:
                continue  # discard low-confidence
            y_true.append(label)
            y_pred.append(pred_label)
            y_conf.append(conf)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    logging.info(
        f"Validation set: accuracy={acc:.2f}, precision={prec:.2f}, recall={rec:.2f}, f1={f1:.2f}, n={len(y_true)}",
        extra={"pipeline_step": "model_eval"},
    )
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'n': len(y_true)}


class EmotionClassifier:
    def predict(self, text):
        # Placeholder for a fear/greed classifier (stub - add real model or use keyword/heuristics)
        # Return 'fear', 'greed', or 'neutral'
        lower_text = text.lower()
        if 'fear' in lower_text or 'crash' in lower_text or 'panic' in lower_text:
            return 'fear'
        if 'greed' in lower_text or 'rally' in lower_text or 'moon' in lower_text:
            return 'greed'
        return 'neutral'
