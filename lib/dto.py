from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, description="Text for classification")

    class Config:
        schema_extra = {
            "example" : {
                "text" : "Text example for classification"
            }
        }

class PredictionResponse(BaseModel):
    prediction: int = Field(..., ge=0, le=1, description="0 - normal, 1 - toxic")
    probability: float = Field(..., ge=0, le=1.0, description="Probability of being toxic")
    confidence: float = Field(..., ge=0, le=1.0, description="Model's confidence")

    class Config:
        schema_extra = {
            "example" : {
                "prediction" : 1,
                "probability" : 0.89,
                "confidence" : 0.92,
            }
    }
    

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=500, description="Texts for classification")
    
    class Config:
        schema_extra = {
            "example" : {
               "texts" : [
                   "First text",
                   "Second text",
                   "So on"
               ]
            }
        }

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

    class Config:
        schema_extra = {
            "example" : {
               "predictions": [
                    {
                        "prediction": 0,
                        "probability": 0.12,
                        "confidence": 0.88
                    },
                    {
                        "prediction": 1,
                        "probability": 0.95,
                        "confidence": 0.95
                    }
                ]
            }
        }


