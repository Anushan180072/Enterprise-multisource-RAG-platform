import logging
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from typing import Optional
from chatbot.logs import logger
from dotenv import load_dotenv
import os
load_dotenv()

class MongoDBClient:
    _client_instance: Optional[MongoClient] = None
    
    def get_client(mongo_uri: str) -> MongoClient:
        """
        Retrieves a singleton instance of the MongoDB client. If no instance exists,
        it creates a new one using the provided MongoDB URI.

        Args:
            mongo_uri (str): The MongoDB connection URI.

        Returns:
            MongoClient: The MongoDB client instance.

        Raises:
            ServerSelectionTimeoutError: If the connection to MongoDB fails.
        """
        if MongoDBClient._client_instance is None:
            try:
                logger.info("Creating a new MongoDB client instance.")
                MongoDBClient._client_instance = MongoClient(mongo_uri)
                #MongoDBClient._client_instance.admin.command('ping')
                logger.info("Successfully connected to MongoDB.")
            except ServerSelectionTimeoutError as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
            
        else:
            logger.info("Using existing MongoDB client instance.")
        return MongoDBClient._client_instance


# MONGO_URI=os.getenv("MONGO_URI") 
# if not MONGO_URI:
#     logger.error("MongoDB URI not found in environment variables.")
    
# else:
#     logger.info("Mongo URl found and connecting the database.")
#     client = MongoDBClient.get_client(MONGO_URI)