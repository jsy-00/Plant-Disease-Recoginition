ARTIFACTS_DIR: str = "artifacts"

DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_DOWNLOAD_URL: str = "https://drive.google.com/file/d/1SiqUfvoOJ6pT8wYKD40SPZZaL6jp5aar/view?usp=sharing"

"""
Data Validation
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_STATUS_FILE = 'status.txt'

DATA_VALIDATION_ALL_REQUIRED_FILES = ["train", "valid","test"]

"""
Model Trainer
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

MODEL_TRAINER_NO_EPOCHS: int = 100

MODEL_TRAINER_BATCH_SIZE: int = 256