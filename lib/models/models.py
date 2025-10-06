import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple, Dict, Any, List
import numpy as np

from utils.logger import logger  


class BertClassifier:
    def __init__(self, model_name: str = "cointegrated/rubert-tiny2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.tokenizer = None
        self.model = None
        
        logger.info(f"Initializing model: {model_name}")
        self._load_model()
        
    def _load_model(self):
        try:
            logger.info(f"Loading model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2, 
            ignore_mismatched_sizes=True)

            self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model {self.model_name} loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Something went wrong during loading model {self.model_name} \
                        on {self.device}: {e}")
            raise
            
    def predict(self, text: str) -> Tuple[int, float, float]:
        try:
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )

            inputs = {key:value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            probs = predictions.cpu().numpy()[0]
            
            prediction = int(np.argmax(probs))
            probability = float(probs[1]) 
            confidence = float(np.max(probs))  
            
            logger.debug(f"Prediction: {prediction}, Probability: {probability:.3f}, Confidence: {confidence:.3f}")
            
            return prediction, probability, confidence
        
        except Exception as e:
            logger.error(f"Something went wrong during prediction: {e}")
            raise


    def predict_batch(self, texts: List[str]) -> List[Tuple[int, float, float]]:
        try:
            inputs = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )

            inputs = {key:value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            probs = predictions.cpu().numpy() 
            results = []

            for i in range(len(texts)):
                prediction = np.argmax(probs[i])
                probability = float(probs[i][1])
                confidence = float(np.max(probs[i]))

                results.append((prediction, probability, confidence))

            return results
        
        except Exception as e:
            logger.error(f"Something went wrong during batch prediction: {e}")
            raise

class ModelManager:
    def __init__(self, classifier: BertClassifier = None):
        self.classifier = classifier or BertClassifier()

        logger.info("Model manager has been initialized")

    def predict(self, text : str) -> Dict[str, Any]:
        prediction, probability, confidence = self.classifier.predict(text)

        return {"prediction" : prediction, 
                "probability" : probability, 
                "confidence" : confidence}
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        results = self.classifier.predict_batch(texts)

        return [
            {
                "prediction" : pred,
                "probability" : prob,
                "confidence" : conf
            }
            for pred, prob, conf in results
        ]