from contextlib import asynccontextmanager


from fastapi import FastAPI


from dependencies import services
from utils.logger import logger 
from handlers import router


    
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API Server")

    services.initialize()
    yield

    logger.info("Shutting down API Server")


app = FastAPI(
    title="Russian Text Toxicity Classificator API",
    description="API for classifiction russin toxic text by rubert-tiny2",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router=router)
