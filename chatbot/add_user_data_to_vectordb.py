from typing import List,Dict,Optional
from langchain.schema import Document
from pymongo import MongoClient
from bson.objectid import ObjectId
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chatbot.data_extract import process_s3_urls
import getpass, os, pymongo, pprint
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_transformers import LongContextReorder
reordering = LongContextReorder()
import re
from pydantic import BaseModel,Field
from fastapi import FastAPI,HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from urllib.parse import urlparse
from typing import List, Tuple
from chatbot.Get_data_from_web import extract_text_from_urls
import asyncio
from chatbot.Get_data_from_Youtube import get_channel_videos_links,get_channel_id,process_youtube_links
from chatbot.logs import logger
import requests
from fastapi import BackgroundTasks
from dotenv import load_dotenv
import atexit
import threading
from pymongo.change_stream import ChangeStream

# Load environment variables from .env file
load_dotenv()

# Access OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key, dimensions=768)
llm = ChatOpenAI(model_name="gpt-4", api_key=openai_api_key,temperature=0,max_retries=2)
#from chatbot.chat_bot import ChatHistoryManager

# Access environment variables

MONGODB_LOCAL_CONNECTION_STRING = os.getenv("MONGODB_LOCAL_CONNECTION_STRING")    
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# DATABASE_NAME = os.getenv("DATABASE_NAME")
# # VECTOR_STORE_DB_NAME = os.getenv("VECTOR_STORE_DB_NAME")
# COLLECTION_NAME = os.getenv("COLLECTION_NAME")
# VECTOR_STORE_COLLECTION_NAME = os.getenv("VECTOR_STORE_COLLECTION_NAME")

# MongoDB Connection
# client = MongoClient(MONGODB_LOCAL_CONNECTION_STRING, tls=True)
# database_name = DATABASE_NAME
# collection_name = COLLECTION_NAME
# collection = client[database_name][collection_name]

# MongoDB Connection Manager (Singleton)
class MongoDBConnectionManager:
    """
    Singleton class to manage MongoDB connections.
    Ensures a persistent connection that does not disconnect after use.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBConnectionManager, cls).__new__(cls)
            cls._instance._init_connection()
        return cls._instance

    def _init_connection(self):
        """
        Initializes the MongoDB connection and keeps it open.
        """
        try:
            self.MONGODB_LOCAL_CONNECTION_STRING = os.getenv("MONGODB_LOCAL_CONNECTION_STRING")
            self.DATABASE_NAME = os.getenv("DATABASE_NAME")
            self.COLLECTION_NAME = os.getenv("COLLECTION_NAME")

            if not self.MONGODB_LOCAL_CONNECTION_STRING:
                raise ValueError("MongoDB connection string is missing in environment variables.")

            # Establish connection
            self.client = MongoClient(self.MONGODB_LOCAL_CONNECTION_STRING, tls=True)
            self.database = self.client[self.DATABASE_NAME]
            self.collection = self.database[self.COLLECTION_NAME]

            logger.info("Successfully connected to MongoDB.")

        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise


    def get_collection(self):
        """
        Returns the MongoDB collection instance.
        """
        return self.collection

    def close_connection(self):
        """
        Closes the MongoDB connection explicitly.
        """
        if self.client:
            self.client.close()
            self._instance = None
            logger.info("MongoDB connection closed.")

# Initialize MongoDB Connection
mongo_manager = MongoDBConnectionManager()
collection = mongo_manager.get_collection()

# Ensure connection is closed when application exits
atexit.register(mongo_manager.close_connection)


class SingletonMeta(type):
    """
    A Singleton metaclass that creates only one instance of a class.
    Here we are using to create the single instance to database connection and loading the embedding model
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class EmbeddingModelManager(metaclass=SingletonMeta):
    """
    Manages the initialization and retrieval of the HuggingFace BGE Embedding model.
    """

    def __init__(self):
        #self.model_name = "BAAI/bge-base-en-v1.5"
        #self.model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        #self.encode_kwargs = {"normalize_embeddings": True}
        #self.embedding_model = None
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        self.embedding_model = None

    def get_embedding_model(self) -> OpenAIEmbeddings:                 #HuggingFaceEmbeddings:  
        """
        Returns the HuggingFace BGE Embedding model instance, initializing it if necessary.
        
        Returns:
            HuggingFaceEmbeddings: The embedding model instance.
        """
        if self.embedding_model is None:
            #self.embedding_model = HuggingFaceEmbeddings(
                #model_name=self.model_name,
                #model_kwargs=self.model_kwargs,
                #encode_kwargs=self.encode_kwargs
            #)
            self.embedding_model = OpenAIEmbeddings(
                model="text-embedding-3-small",  # text-embedding-3-large
                api_key=self.api_key,
                #show_progress_bar=True,
                dimensions=768,
                request_timeout=30,
                chunk_size=300
            )
        return self.embedding_model
    


class VectorStoreManager(metaclass=SingletonMeta):
    """
    Manages the initialization and retrieval of the MongoDB Atlas Vector Store.
    """

    def __init__(self):
        self.ATLAS_CONNECTION_STRING = os.getenv("AI_CONNECTION_STRING")
        if not self.ATLAS_CONNECTION_STRING:
            raise ValueError("AI_CONNECTION_STRING is not set in .env file")

        self.client = MongoClient(self.ATLAS_CONNECTION_STRING)
        self.db_name = "chatbot"
        self.collection_name = "user_files"
        self.vector_store = None

    def get_vector_store(self) -> MongoDBAtlasVectorSearch:
        """
        Returns the MongoDB Atlas Vector Store instance, initializing it if necessary.
        
        Returns:
            MongoDBAtlasVectorSearch: The vector store instance.
        """
        if self.vector_store is None:
            atlas_collection = self.client[self.db_name][self.collection_name]
            embedding_manager = EmbeddingModelManager()
            self.vector_store = MongoDBAtlasVectorSearch(
                embedding=embedding_manager.get_embedding_model(),
                #embedding=openai_embeddings,
                collection=atlas_collection
            )
        return self.vector_store,atlas_collection   
    
vector_store_manager = VectorStoreManager()
vector_store,atlas_collection= vector_store_manager.get_vector_store()

def get_text_chunks(documents: List[Document]) -> List[Document]:

    """
    Splits the text in a list of LangChain documents into smaller chunks.

    Args:
        documents (List[Document]): A list of LangChain documents to be split.

    Returns:
        List[Document]: A list of LangChain documents with text split into chunks.
    """

    logger.info("Splitting documents into chunks.")
    try:

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        #print("\ntextsplit:",text_splitter)
        chunks = text_splitter.split_documents(documents)
        #print("\nchunkking:",chunks)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting the documents into chunks :{e}")
        raise

def add_admin_id_to_docs(documents: List[Document], admin_id: str, support_email: Optional[str] = None) -> List[Document]:
    """
    Adds admin ID, support email to metadata, and embeds the source into the document content.

    Args:
        documents (List[Document]): A list of LangChain documents.
        admin_id (str): Admin ID to be added to metadata.
        support_email (Optional[str]): Optional support email to be added to metadata.

    Returns:
        List[Document]: List of updated documents.
    """
    logger.debug("Adding admin ID and support email to documents and embedding source into text.")

    try:
        updated_docs = []
        for doc in documents:
            # Add metadata
            doc.metadata['admin_id'] = admin_id
            if support_email:
                doc.metadata['support_email'] = support_email

            source = doc.metadata.get('source')
            if source:
                doc.page_content = f"{source}\n\n{doc.page_content}"
                logger.debug(f"Embedded source into document text: {source}")

            updated_docs.append(doc)

        logger.info(f"Added metadata and updated content for {len(updated_docs)} documents.")
        return updated_docs

    except Exception as e:
        logger.error(f"Error adding admin ID '{admin_id}' to documents: {e}")
        raise

def preprocess_documents(docs: List[Document]) -> List[Document]:
    """
    Cleans the text of each document by removing extra newlines and excessive whitespace.

    Args:
        docs (List[Document]): A list of LangChain documents to be cleaned.

    Returns:
        List[Document]: A list of LangChain documents with cleaned text.

    """
    logger.debug("Preprocessing/cleaning the documents")
    try:

        cleaned_docs = []
        for doc in docs:
            cleaned_text = re.sub(r'\n{2,}', '\n', doc.page_content)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            cleaned_doc = Document(page_content=cleaned_text, metadata=doc.metadata)
            cleaned_docs.append(cleaned_doc)
        logger.info(f"preprocessed {len(doc)} documents.")
        return cleaned_docs
    except Exception as e:
        logger.error(f"Error preprocessing the documents:{e}")
        raise




def classify_urls(urls: List[str]) -> Tuple[List[str], List[str]]:
    """
    Classifies a list of URLs into YouTube URLs and generic website URLs.

    Parameters:
    - urls (List[str]): A list of URLs to classify.

    Returns:
    - Tuple[List[str], List[str]]: Two lists, one with YouTube URLs and the other with website URLs.
    """
    logger.info("Classifying the URL into the Youtube and the website Urls.")
    youtube_domains = {'www.youtube.com', 'youtube.com', 'm.youtube.com', 'youtu.be'}
    youtube_urls = []
    website_urls = []

    for url in urls:
        parsed_url = urlparse(url)
        if parsed_url.netloc in youtube_domains or parsed_url.path.startswith('@'):
            #print(parsed_url.netloc)
            youtube_urls.append(url)
        else:
            website_urls.append(url)
    logger.info(f"Classified {len(urls)} URLs: {len(youtube_urls)} YouTube URLs and {len(website_urls)} website URLs.")
    return youtube_urls, website_urls


def data_from_youtube_web(urls: List[str])-> list[Document]:
    """
    This function retrieves data from YouTube and websites. It calls other functions internally 
    to process and return the data.

    Parameters:
    - urls (List[str]): A list of URLs (both web and YouTube) to process.
    
    Returns:
    - List[Document]: A list of LangChain Document objects containing the extracted data.
    """
    logger.info("Getting data from the Youtube and websites")

    try:

        youtube_urls, website_urls=classify_urls(urls)
        data=[]

        logger.info(f"Processing the website url")
        try:
            web_data=extract_text_from_urls(website_urls)
            data.extend(web_data)
        except Exception as e:
            logger.error(f"Failed to getting the data from {website_urls} and error is {e}")
            raise

        logger.info("Processing the Youtube Urls")
        try:

            for youtubeurl in youtube_urls:
                if youtubeurl.startswith("@"):
                    channel_id=get_channel_id(youtubeurl)
                    channel_links=get_channel_videos_links(channel_id)
                    youtube_data=process_youtube_links(channel_links)
                    data.extend(youtube_data)
                else:
                    youtube_data1=process_youtube_links([youtubeurl])
                    #print(youtube_data1)
                    data.extend(youtube_data1)

            return data
        except Exception as e:
            logger.error(f"Failed to retrieve the data from Youtube Url  {e}")
            raise

    except Exception as e:
        logger.error(f"Error Processig  urls {urls} and error is {e}")
        raise
    
    # Recursive function to convert nested dictionary into a string
def dict_to_string(data, level=0) -> str:
    indent = '  ' * level  # Indentation for each level of nesting
    output = ""
    
    if isinstance(data, dict):
        for key, value in data.items():
            output += f"{indent}{key}:"
            if isinstance(value, (dict, list)):
                output += "\n" + dict_to_string(value, level + 1)  # Recurse for nested dicts/lists
            else:
                output += f" {value}\n"
    elif isinstance(data, list):
        for item in data:
            output += f"{indent}-\n{dict_to_string(item, level + 1)}"
    else:
        output += f" {data}\n"
    
    return output
# Fetch Data Using entityId
def fetch_data_by_entity(entityId: list) -> list:
    """
    Fetches data for multiple entities from the database.

    Args:
        entityIds (list): A list of entity IDs.

    Returns:
        list: A list of aggregated data for the provided entity IDs.
    """
    pipeline = [
        {"$match": {"entity_id": {"$in": [ObjectId(eid) for eid in entityId]}, "status": "ACTIVE"}},
        {"$unwind": "$templates_data"},
        {
            "$lookup": {
                "from": "templates_data",
                "localField": "templates_data.template_data_id",
                "foreignField": "_id",
                "as": "template_details",
            }
        },
        {"$unwind": "$template_details"},
        {
            "$group": {
                "_id": "$entity_id",
                "name": {"$first": "$name"},
                "templates_data": {
                    "$push": {
                        "template_data_id": "$templates_data.template_data_id",
                        "template_details": "$template_details.template_data",
                    }
                },
            }
        },
    ]
    result = list(collection.aggregate(pipeline))
    return result

def convert_to_documents(data: list) -> list:
    """
    Converts a list of entity data into multiple LangChain Document objects.
    
    Args:
        data (list): The list of entities and their templates.

    Returns:
        list: A list of Document objects, where each template is a separate document.
    """
    documents = []
    
    for entity in data:
        # entity_id = entity.get('_id')
        name = entity.get('name', 'Unknown')

        for template in entity.get('templates_data', []):
            template_id = template.get('template_data_id', 'Unknown')
            template_details = dict_to_string(template.get('template_details', {}))
            
            doc_content = f"{template_id}\n{template_details}"
            
            # Create a separate document for each template
            documents.append(Document(page_content=doc_content.strip()))
    
    return documents
        
def process_entity(entityId: list):
    """
    Processes a list of entity IDs by fetching data, converting it to multiple documents,
    and splitting them into smaller chunks.
    """
    try:
        # Fetch data for the given entity IDs
        data_from_entity = fetch_data_by_entity(entityId)
        if not data_from_entity:
            print(f"No data found for entity ID: {entityId}")
            return

        # Convert fetched data into multiple separate documents
        documents = convert_to_documents(data_from_entity)
        #print("\nConverted Documents:", documents)

        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        # Split each document into chunks
        chunks = text_splitter.split_documents(documents)
        #print("\nSplit Chunks:", chunks)

        # Output chunks
        for i, chunk in enumerate(chunks, start=1):
            print(f"Chunk {i}:\n{chunk.page_content}\n")

    except Exception as e:
        print(f"An error occurred: {e}")

def activate_bot(bot_id: str):
    """
    Activates a bot by sending a POST request to the specified API.

    Args:
        bot_id (str): The ID of the bot to activate.

    Returns:
        dict: The JSON response from the API if successful, None otherwise.
    """
    url = 'https://ai-bot.kodefast.com/active_bot.php'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'id': bot_id
    }
    
    logger.info(f"Attempting to activate bot with ID: {bot_id}")
    
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Bot activated successfully. Response: {result}")
        return result
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while activating the bot: {http_err}")
    except Exception as err:
        logger.error(f"An error occurred while activating the chatbot of user {bot_id}: {err}")



def store_user_all_data_with_id(urls: List[str], id: str,file_url: List[str],support_email: Optional[str]=None) -> Dict:
    """
    Process a list of S3 URLs and add metadata including admin_id.

    Args:
        urls (List[str]): List of S3 URLs.
        admin_id (str): Admin ID to add as metadata.
        support_email (str): Support email for user.

    Returns:
        Dict: A dictionary with the processed results.
    """
    logger.info(f"Storing the user data with admin ID:{id}")
    try:
        final_user_data=[]
        logger.info("Processing S3 URLs.")
        data_from_s3_url=process_s3_urls(file_url)
        final_user_data.extend(data_from_s3_url)
        #print("\n\nfinal data:",final_user_data)
        logger.info("Processe S3 URLs successfully.")

        data_from_web_youtube=data_from_youtube_web(urls)
        final_user_data.extend(data_from_web_youtube)
        #print("\n\nweb:",data_from_web_youtube)

        #print(data_from_url)
        chunked_data=get_text_chunks(final_user_data)
        #print("\nchunn:",chunked_data)
        id_added_data=add_admin_id_to_docs(chunked_data,id,support_email)
        # #cleaned_data=preprocess_documents(add_admin_id_to_docs)
        vector_store.add_documents(id_added_data)
        logger.info("User data successfully stored in the vector store.")
        return True
    except Exception as e:
         logger.error(f"Error storing user data with ID {id}: {e}")
         return False
        
def store_user_all_data_with(entityId: Optional[List[str]], id: str, support_email: Optional[str] = None) -> Dict:
    """
    Process a list of entity IDs and store them as separate documents with metadata.
    """
    logger.info(f"Storing user data with admin ID: {id}")

    try:
        # Fetch and convert data to multiple documents
        data_from_entity = fetch_data_by_entity(entityId)
        if not data_from_entity:
            logger.warning(f"No data found for entity ID: {entityId}")
            return False
        
        documents = convert_to_documents(data_from_entity)

        # Add metadata (admin ID & support email)
        id_add_data = add_admin_id_to_docs(documents, id, support_email)

        # Store each document separately in the vector database
        vector_store.add_documents(id_add_data)

        logger.info(f"Successfully stored {len(id_add_data)} documents in the vector store.")
        return True
    except Exception as e:
        logger.error(f"Error storing user data with ID {id}: {e}")
        return False

class APIRequest(BaseModel):
    id: Optional[str] = Field(None, description="Unique identifier for the user")
    primary_use_case: Optional[str] = Field(None, description="Primary use case for the data")
    industry: Optional[str] = Field(None, description="Industry related to the data")
    role: Optional[str] = Field(None, description="Role related to the data")
    has_website: Optional[str] = Field(None, description="Indicates if there is a website")
    website: Optional[str] = Field(None, description="Website URL")
    bot_name: Optional[str] = Field(None, description="Name of the bot")
    support_email: Optional[str] = Field(None, description="Support Email for the bot")
    welcome_message: Optional[str] = Field(None, description="Welcome message for the bot")
    suggested_reply: Optional[str] = Field(None, description="Suggested reply from the bot")
    entityId: Optional[List[str]]=Field(None,description="List of entities")
    urls: Optional[List[str]] = Field(None, description="List of URLs including S3 bucket URLs")
    fileUrls: Optional[List[str]] = Field(None, description="List of file download URLs")

app9= FastAPI()



def data_ingestion_background(input:APIRequest):
    """
    Processes the data provided in the API request, including storing the user data and activating the bot.

    Args:
        input (APIRequest): The input data containing user information, URLs, and other relevant details.

    Raises:
        Exception: If any error occurs during data processing.
    """
    try:
        logger.info(f"processing data for user ID: {input.id}")
        if store_user_all_data_with(input.entityId,input.id):
            logger.info(f"User data successfully stored of botid {input.id}")
            activate_bot(input.id)
         
        elif store_user_all_data_with_id(input.urls,input.id,input.fileUrls,input.support_email):
            logger.info(f"User data successfully stored of botid {input.id}")
            activate_bot(input.id)
        else:
            logger.error(f" data ingestion failed {input.id}")
            raise HTTPException(status_code=500, detail="Data ingestion` failed.")
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Error ingestion data: {str(e)}")




@app9.post("/")
async def ingest_data(input: APIRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to ingest data from a list of S3, web, and YouTube URLs along with an admin ID.

    This endpoint immediately acknowledges the receipt of the request and processes the data in the background.
    
    Args:
        input (APIRequest): Input containing user information, URLs, and other relevant details.
        background_tasks (BackgroundTasks): FastAPI BackgroundTasks instance for executing tasks in the background.

    Returns:
        dict: A message confirming that the request has been received and is being processed in the background.
    """
    logger.info(f"Received data ingestion request for user ID: {input.id}")
    background_tasks.add_task(data_ingestion_background, input)
    return {'success': True, 'message': 'API data received.'}


from chatbot.get_chat_histroy import get_all_bots_chat,bot_statistics
from datetime import datetime
class GetChatRequest(BaseModel):
    admin_ids:Optional[list[str]]=Field(None,description="list of admin_id (bot_id) to get the chathistory with in the given date range")
    start_date: Optional[datetime] = Field(None, description="The start time for filtering chat history")
    end_date: Optional[datetime] = Field(None, description="The end time for filtering chat history")

@app9.post("/get_chat")
async def get_chat(input:GetChatRequest):
    logger.debug("starting point In API to get chat_history")
    try:
        chatHistroy=get_all_bots_chat(input.admin_ids,input.start_date,input.end_date)
        stats=bot_statistics(chatHistroy)
        return chatHistroy,stats
    except Exception as e:
        logger.error("Error getting the chat_histroy of admin id")
        raise HTTPException(status_code=500,detail=f"Error Processing:{str(e)}")
    
    
from chatbot.get_chat_histroy import delete_sessions
class DeleteRequest(BaseModel):
    admin_id: str
    session_id: str

@app9.delete("/delete_session")
async def delete_session(input:DeleteRequest):
    logger.debug("starting point in API to delete session")
    try:
        success, deleted_count = delete_sessions(input.admin_id, input.session_id)
        if success and deleted_count>0:
            return {'success': True, 'message': f"Session {input.session_id} with admin_id {input.admin_id} deleted successfully"}
        elif success and deleted_count<=0:
            return {'success': False, 'message': f"No session found with session_id {input.session_id} and admin_id {input.admin_id}"}
        else:
            return {'success': False, 'message': f"Error occurred while deleting the session {input.session_id} with admin_id {input.admin_id}"}
    except Exception as e:
        logger.error("Error occurred while deleting the session")
        raise HTTPException(status_code=500, detail=f"Error Processing:{str(e)}")
    