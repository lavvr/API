from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from dto import PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from clients import APIClient
from processors import TextPreprocessor
from models import BertClassifier, ModelManager
from utils.logger import logger 


text_preprocessor = None
bert_classifier = None
api_client = None
model_manager = None

def initialize_services():
    global text_preprocessor, bert_classifier, api_client, model_manager

    logger.info("Initializing services")

    try:
        text_preprocessor = TextPreprocessor(remove_commas=True, min_word_length=2)
    except Exception as e:
        logger.error(f"Something went wrong during preprocessor initialization: {e}")
        raise

    try:
        api_client = APIClient(text_preprocessor)
    except Exception as e:
        logger.error(f"Something went wrong during API Client initialization: {e}")
        raise
       
    try:
        bert_classifier = BertClassifier()
    except Exception as e:
        logger.error(f"Something went wrong during BERT Classifier initilization: {e}")

    try:
        model_manager = ModelManager(classifier=BertClassifier())
    except Exception as e:
        logger.error(f"Something went wrong during Model Manager initilization: {e}")

    
@asynccontextmanager
async def lifespan():
    logger.info("Starting API Server")

    initialize_services()
    yield

    logger.info("Shutting down API Server")


app = FastAPI(
    title="Russian Text Toxicity Classificator API",
    description="API for classifiction russin toxic text by rubert-tiny2",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {
        "message" : "Russian Text Toxicity Classification API",
        "status" : "running",
        "model" : "cointegrated/rubert-tiny2"
    }

@app.get("/health")
async def health_check():
    health_status = {
        "status" : "healthy",
        "model_loaded" : bert_classifier is not None and bert_classifier.model is not None
    }

    logger.debug(f"Health status : {health_status}")
    return health_status

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    logger.info(f"Predict endpoint called with text: {request.text[:100]}...")

    try:
        processed_text = api_client.process_request(request.text)
        result = model_manager.predict(processed_text)

        logger.info(f"Prediction completed: {result}")
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Error in prediction endpoint")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/batch_predict", response_model=BatchPredictionRequest)
async def batch_predict(request: BatchPredictionResponse):
    logger.info(f"Batch Predict endpoint called with {len(request.texts)} texts.")

    try:
        if len(request.texts) > 500:
            logger.warning(f"Too many texts, max is 500.\n Received:{len(request.texts)}")
            raise HTTPException(status_code=400)
        
        processed_texts = api_client.process_batch(request.texts)
        results = model_manager.predict_batch(processed_texts)

        prediction_responses = [PredictionResponse(**results) for result in results]

        logger.info(f"Batch prediction completed, predicted {len(request.texts)} texts")
        return BatchPredictionResponse(prediction=prediction_responses)
    
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint.")
        raise HTTPException(status_code=500, detail=str(e))

