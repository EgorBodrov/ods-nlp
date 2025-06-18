from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re


import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')


import warnings
warnings.filterwarnings("ignore")


class DebertaV3:
    def __init__(self, model_path: str, device: int = 1):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path).to(device)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.device = device
        self._lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text):
        text = re.sub(r'<br\s*/?>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.lower()
        tokens = text.split()

        tokens = [self._lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def predict(self, text: str):
        preprocessed = self.preprocess_text(text)
        inputs = self.tokenizer(
            preprocessed,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
        return {
            "label": predicted_class,
            "confidence": probs[0][predicted_class].item()
        }
