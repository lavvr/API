import logging

class ApiLogger:
    def __init__(self,
                 name='app',
                 log_level='INFO',
                 log_to_console=True,
                 log_to_file=False,
                 log_file_path='app.log'):
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level))

        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            if log_to_console:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

            if log_to_file:
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)
    
    def exception(self, message):
        self.logger.exception(message)


logger = ApiLogger()