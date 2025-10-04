import re
from typing import List


from utils.logger import logger


class TextPreProcessor:

    def __init__(self, remove_commas: bool = True, min_word_length: int = 2):        
        self.remove_commas = remove_commas
        self.min_word_length = min_word_length

        logger.info(f"Text preproccesor has been initialized: \n \
                    remove_commas: {self.remove_commas}, min_word_length: \
        {self.min_word_length}")

    def preprocess(self, text: str) -> str:
        
        if not isinstance(text, str):
            logger.error("Input text must be a string")

        if self.remove_commas:
            text = text.replace(',', ' ')

        words = text.split(' ')
        filtered_words = [word for word in words if len(word.strip()) >= self.min_word_length 
                          or not word.strip().isalpha()]
        
        processed_text = ' '.join(filtered_words)
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()

        return processed_text