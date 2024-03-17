import sys
from src.Project.logger import logging
from src.Project.exception import CustomException

if __name__ == '__main__':
    logging.info("The execution has started")
    try:
        a = 1/0
    
    except Exception as e:
        logging.info("Custom Exception Created")
        raise CustomException(e,sys)