from typing import List

from processors import TextPreprocessor
from utils.logger import logger


class APIClient:
    def __init__(self, preprocessor: TextPreprocessor | None = None):
        self.preprocessor = preprocessor or TextPreprocessor()

        logger.info("APIClient has been initialized")

    def process_request(self, text: str) -> str:

        try:
            if not text:
                logger.warning("Text cannot be empty")
                raise ValueError
            return self.preprocessor.preprocess(text)
        
        except Exception as e:
            logger.error(f"Something went wrong: {e}")
            raise

    def process_batch(self, texts: List[str]) -> List[str]:
        return [self.preprocessor.preprocess(text) for text in texts]
        