from fastapi import APIRouter, HTTPException


from dependencies.dependencies import services
from lib.dto import PredictionRequest, PredictionResponse, BatchPredictionRequest, BatchPredictionResponse
from utils.logger import logger


router = APIRouter()

@router.get("/")
async def root():
    logger.info("Root endpoint called")
    return {
        "message" : "Russian Text Toxicity Classification API",
        "status" : "running",
        "model" : "cointegrated/rubert-tiny2"
    }

@router.get("/health")
async def health_check():
    health_status = {
        "status" : "healthy",
        "model_loaded" : services.bert_classifier is not None and services.bert_classifier.model is not None
    }

    logger.debug(f"Health status : {health_status}")
    return health_status

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    logger.info(f"Predict endpoint called with text: {request.text[:100]}...")

    try:
        processed_text = services.api_client.process_request(request.text)
        result = services.model_manager.predict(processed_text)

        logger.info(f"Prediction completed: {result}")
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Error in prediction endpoint")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    logger.info(f"Batch Predict endpoint called with {len(request.texts)} texts.")

    try:
        if len(request.texts) > 500:
            logger.warning(f"Too many texts, max is 500.\n Received:{len(request.texts)}")
            raise HTTPException(status_code=400)
        
        processed_texts = services.api_client.process_batch(request.texts)
        results = services.model_manager.predict_batch(processed_texts)

        prediction_responses = [PredictionResponse(**result) for result in results]

        logger.info(f"Batch prediction completed, predicted {len(request.texts)} texts")
        return BatchPredictionResponse(predictions=prediction_responses)
    
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint.")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def root():
    logger.info("Root endpoint called")
    return {
        "message" : "Russian Text Toxicity Classification API",
        "status" : "running",
        "model" : "cointegrated/rubert-tiny2"
    }

@router.get("/health")
async def health_check():
    health_status = {
        "status" : "healthy",
        "model_loaded" : services.bert_classifier is not None and services.bert_classifier.model is not None
    }

    logger.debug(f"Health status : {health_status}")
    return health_status

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    logger.info(f"Predict endpoint called with text: {request.text[:100]}...")

    try:
        processed_text = services.api_client.process_request(request.text)
        result = services.model_manager.predict(processed_text)

        logger.info(f"Prediction completed: {result}")
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Error in prediction endpoint")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    logger.info(f"Batch Predict endpoint called with {len(request.texts)} texts.")

    try:
        if len(request.texts) > 500:
            logger.warning(f"Too many texts, max is 500.\n Received:{len(request.texts)}")
            raise HTTPException(status_code=400)
        
        processed_texts = services.api_client.process_batch(request.texts)
        results = services.model_manager.predict_batch(processed_texts)

        prediction_responses = [PredictionResponse(**result) for result in results]

        logger.info(f"Batch prediction completed, predicted {len(request.texts)} texts")
        return BatchPredictionResponse(predictions=prediction_responses)
    
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint.")
        raise HTTPException(status_code=500, detail=str(e))

