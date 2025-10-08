from lib.clients import APIClient
from lib.processors import TextPreprocessor
from lib.models import BertClassifier, ModelManager
from lib.utils import logger


class Service():
    def __init__(self):
        self.text_preprocessor = None
        self.bert_classifier = None
        self.api_client = None
        self.model_manager = None

        self._initialized = False

    def initialize(self):

        if self._initialized:
            return
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
        model_manager = ModelManager(classifier=bert_classifier)
    except Exception as e:
        logger.error(f"Something went wrong during Model Manager initilization: {e}")

    logger.info("Services initialized")

services = Service()