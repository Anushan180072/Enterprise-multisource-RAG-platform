import logging
from logging.handlers import RotatingFileHandler
import os
from dotenv import load_dotenv
import pymongo
from pymongo import MongoClient
from datetime import datetime
import time

load_dotenv()
# MONGO_URI = os.getenv("MONGO_URI")
# DB_NAME = os.getenv("DB_NAME")
# COLLECTION_NAME = os.getenv("COLLECTION_NAME")
# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# collection = db[COLLECTION_NAME]

# Define the directory and log file name
log_directory = 'chatbot'
log_file = 'chatbot.logs'
log_path = os.path.join(log_directory, log_file)

def setup_logging():
    """Configure and return a logger instance with rotation."""
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    handler = RotatingFileHandler(
        log_path, 
        maxBytes=2*1024*1024,  
        backupCount=5 
    )
    
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            handler,
            logging.StreamHandler()  # Output logs to the console as well
        ]
    )
    
    return logging.getLogger(__name__)

logger = setup_logging()

logger.info("testing logging system")



# def store_logs_in_mongodb():
#     """Store the log file contents in MongoDB."""
#     try:
#         logger.info("Storing logs in MongoDB.")
#         with open(log_file, 'r') as file:
#             log_contents = file.read()
        
#         log_entry = {
#             'timestamp': datetime.now(),
#             'log_file': log_file,
#             'content': log_contents
#         }
#         collection.insert_one(log_entry)
#         logger.info("Logs stored in MongoDB successfully.")
#     except Exception as e:
#         logger.error(f"Failed to store logs in MongoDB: {e}")

