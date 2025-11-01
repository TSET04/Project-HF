from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class FinBertSentimentAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().numpy()[0]
        sentiment_score = probs[2] - probs[0] # positive - negative
        return float(sentiment_score)

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
