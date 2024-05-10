import sys
from src.Project.logger import logging
from src.Project.exception import CustomException
#from src.Project.components.model_trainer import modeltrainer
from src.Project.pipeline.training_pipeline import TrainPipeline

"""if __name__ == '__main__':
    logging.info("The execution has started")
    try:
        model = modeltrainer()
        train_dir = r"artifact\Data\train"
        val_dir = r"artifact\Data\valid"
        batch_size = 128
        epoch = 30
        model.initiatemodel(train_dir, val_dir, batch_size, epoch)
        print(model.get_classlabels)
        logging.info("Model Training has completed")
        
    except Exception as e:
        raise CustomException(e,sys)"""

obj = TrainPipeline() 
obj.run_pipeline()