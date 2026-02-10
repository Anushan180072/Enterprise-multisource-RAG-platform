# from langchain.schema import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores.utils import filter_complex_metadata
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import Chroma
# import warnings
# from pydantic import BaseModel
# from PyPDF2 import PdfReader
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.chains import ConversationChain
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredExcelLoader, PyPDFLoader, UnstructuredPowerPointLoader, CSVLoader, Docx2txtLoader
# from bs4 import BeautifulSoup
# from langchain_google_genai import GoogleGenerativeAI
# import google.generativeai as genai
# from pptx import Presentation
# import json
# from pathlib import Path
# import logging
# import pandas as pd
# import os
# import tempfile
# #import unstructured
# from PIL import Image
# import mimetypes
# from langchain.memory import ConversationBufferWindowMemory
# from langchain_community.document_loaders import YoutubeLoader
# from langchain_community.document_loaders.sitemap import SitemapLoader
# api_key = "AIzaSyBD_g1FZAOsyPRQvBGwoGM3uxNB3pq38Es"
# genai.configure(api_key=api_key)
# #vector_db1=None
# from langchain_community.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.memory import ConversationBufferWindowMemory
# from langchain.chains import ConversationalRetrievalChain
# import time
# import urllib.parse
# from langchain.retrievers import EnsembleRetriever
# from langchain_community.retrievers import BM25Retriever
# from langchain.chains.question_answering import load_qa_chain
# from langchain_community.document_transformers import LongContextReorder
# reordering=LongContextReorder()
# import shelve
# from langchain.prompts import (
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
#     ChatPromptTemplate,
#     MessagesPlaceholder)
# from fastapi import FastAPI, Form
# from fastapi.responses import JSONResponse
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# #embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=api_key)
# #embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# llm = GoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=api_key, temperature=0, verbose=True)
# api="sk-proj-PtvCwFGTPfrwgtDGQWt5T3BlbkFJl3RO912vwXvqr87S0CVt"
# from langchain.embeddings.openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=api,dimensions=2000)


# ##function to load data from the Multiple Images
# VALID_FORMATS = ['image/jpeg', 'image/png', 'image/gif']
# MODEL_CONFIG = {
#   "temperature": 0.0,
#   "top_p": 1,
#   "top_k": 32,
#   "max_output_tokens": 4096,
# }

# ## Safety Settings of Model
# safety_settings = [
#   {
#     "category": "HARM_CATEGORY_HARASSMENT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_HATE_SPEECH",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   },
#   {
#     "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#     "threshold": "BLOCK_MEDIUM_AND_ABOVE"
#   }
# ]


# # Initialize GenAI GenerativeVision Model for extracting the data from the Images
# model = genai.GenerativeModel(model_name="gemini-pro-vision", generation_config=MODEL_CONFIG, safety_settings=safety_settings)
# from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory

# # Replace the following with your MongoDB connection details
# db_name = "AI-ML"
# collection_name = "chatbot_chat_history"

# def get_user_memory(user_id):
#     return MongoDBChatMessageHistory(
#         connection_string=mongo_uri,
#         database_name=db_name,
#         collection_name=collection_name,
#         session_id=user_id,
#     )

# # Supported image formats
# VALID_FORMATS = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff']

# def upload_images(image_paths):
#     documents = []

#     for image_path in image_paths:
#         try:
#             text_from_image = process_image_and_answer(image_path)
#             documents.append(Document(page_content=text_from_image, metadata={"source": Path(image_path).name}))
#         except Exception as e:
#             logger.error(f"Error processing image {Path(image_path).name}: {e}")

#     return documents

# # Function to process image and generate text
# def process_image_and_answer(image_path):
#     def image_format(image_path):
#         img = Path(image_path)
#         if not img.exists():
#             raise FileNotFoundError(f"Image file not found: {img}")

#         mime_type, _ = mimetypes.guess_type(str(img))
#         if not mime_type or mime_type not in VALID_FORMATS:
#             raise ValueError(f"Invalid image format: {img}")

#         image_parts = [
#             {
#                 'mime_type': mime_type,
#                 'data': img.read_bytes()
#             }
#         ]
#         return image_parts

#     def gemini_output(image_path):
#         image_info = image_format(image_path)
#         response = model.generate_content(image_info)
#         return response.text

#     logger.info(f"Processing image: {image_path}")
#     output = gemini_output(image_path)
#     return output

 


# ### function to load text from youtube video
# def load_websites(url):
#     try:
#         path = url + "sitemap.xml"
#         sitemap_loader = SitemapLoader(web_path=path, continue_on_failure=True)
#         return sitemap_loader.load()
#     except Exception as e:
#         logger.error(f"Error loading website: {e}")
#         return []

# def extract_text(url):
#     try:
#         docs = load_websites(url)
#         if docs:
#             return docs
#     except Exception as e:
#         logger.error(f"Error extracting text from website: {e}")

#     if "youtube.com" in url:
#         try:
#             return YoutubeLoader.from_youtube_url(url, add_video_info=True).load()
#         except ValueError as e:
#             return f"Error processing YouTube URL '{url}': {e}"

#     return f"Unsupported URL '{url}'"

# # Function to load data from Multiples Files file
# def document_data_load(files):
#     documents = []
    
#     # List all files in the given director
#     for file in files:
#         file_extension = file.suffix.lower()

#         try:
#             if file_extension == ".pdf":
#                 loader = PyPDFLoader(str(file))
#                 data = loader.load()
#             elif file_extension == ".pptx":
#                 loader = UnstructuredPowerPointLoader(str(file))
#                 data = loader.load()
#             elif file_extension == ".csv":
#                 loader = CSVLoader(str(file))
#                 data = loader.load()
#             elif file_extension == ".docx":
#                 loader = Docx2txtLoader(str(file))
#                 data = loader.load()
#             elif file_extension == ".txt":
#                 with open(file, 'r') as txt_file:
#                     data = txt_file.read()
#                     documents.append(Document(page_content=data, metadata={"source": file.name}))
#                 continue
#             elif file_extension == ".xlsx":
#                 loader = UnstructuredExcelLoader(str(file), mode="elements")
#                 xlsx_data = loader.load()
#                 html_content = str(xlsx_data)
#                 soup = BeautifulSoup(html_content, 'html.parser')
#                 rows = soup.find('tbody').find_all('tr')
#                 excel_data = []
#                 for row in rows:
#                     cells = row.find_all('td')
#                     excel_data.extend([cell.get_text() for cell in cells])
#                 data = '\n'.join(excel_data)
#             elif file_extension == ".json":
#                 with open(file, 'r') as json_file:
#                     data = json.load(json_file)
#                     documents.append(Document(page_content=json.dumps(data, indent=2), metadata={"source": file.name}))
#                 continue
#             elif file_extension == ".doc":
#                 document = Document()
#                 document.LoadFromFile("/content/sample.doc")
#                 document_text = document.GetText()
#                 documents.append(Document(page_content=document_text, metadata={"source": file.name}))
                
#             else:
#                 raise ValueError(f"Unsupported file type: {file_extension}")

#             # Ensure data is a string for Document page_content
#             if isinstance(data, list):
#                 for item in data:
#                     if isinstance(item, Document):
#                         documents.append(item)
#                     else:
#                         documents.append(Document(page_content=str(item), metadata={"source": file.name}))
#             else:
#                 documents.append(Document(page_content=str(data), metadata={"source": file.name}))

#         except Exception as e:
#             print(f"Error processing file {file.name}: {e}")
    
#     return documents


# def data_from_directory(directory_path):
#     directory = Path(directory_path)
#     if not directory.is_dir():
#         raise NotADirectoryError(f"Provided path is not a directory: {directory_path}")
#     image_extensions = ['.bmp', '.jpeg', '.jpg', '.png', '.gif', '.tiff', '.tif']
#     document_extensions = ['.txt', '.pdf', '.doc', '.docx', '.odt', '.rtf', '.xlsx', '.xls', '.csv', '.pptx', '.ppt', '.html', '.htm', '.json']
#     image_files = []
#     document_files = []
#     for file in directory.iterdir():
#         if file.is_file():
#             if file.suffix.lower() in image_extensions:
#                 image_files.append(file)
#             elif file.suffix.lower() in document_extensions:
#                 document_files.append(file)
#     document_data=document_data_load(document_files)
#     #print("the document data is",document_data)
#     imges_data=upload_images(image_files)
#     #print("the image data is", imges_data)
#     total_data=document_data+imges_data
#     return total_data


  




# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_documents(text)
#     return chunks




# def get_vector_store2(text_chunks,path):
#   try:
      
#     persist_directory =path
#     vectordb = Chroma.from_documents(text_chunks, embeddings,
#       persist_directory=persist_directory)
#     return vectordb
#   except Exception as e:
#         logger.error(f"Error creating vector store: {e}")
#         return None


# def store_doc(text, loc):
#     try:
#         directory = os.path.dirname(loc)
#         if not os.path.exists(directory):
#             os.makedirs(directory)
        
#         existing_text = retrieve_doc(loc)
#         combined_text = existing_text + text if existing_text else text
        
#         with shelve.open(loc) as db:
#             db['document'] = combined_text
#             logger.info('Document stored successfully!')
#     except Exception as e:
#         logger.error(f"Error storing document: {e}")
#     finally:
#         db.close()

# def retrieve_doc(loc):
#     try:
#         with shelve.open(loc) as db:
#             return db.get('document')
#     except Exception as e:
#         logger.error(f"Error retrieving document: {e}")
#         return None
#     finally:
#         db.close()


# def get_conversational_chain(question, user_id,vectors):
#     try:
#         user_specific_memory = get_user_memory(user_id)
#         memory = ConversationBufferWindowMemory(
#             memory_key="chat_history",
#             chat_memory=user_specific_memory,
#             input_key="human_input",
#             k=10
#         )
        
#         #ensemble_retriever is now initialized once and reused
#         ensemble_docs = ensemble_retriever.invoke(question)
#         #print("Entered to the conversational chain")
#         #retriver=vectors.as_retriever()
#         #
#         # docs=retriver.invoke(question)
#         #print("The ensemble docs are", ensemble_docs)
        
#         reordering_docs = reordering.transform_documents(ensemble_docs)

#         template ="""You are an Esigns assistant chatbot named LIca Bot. Your job is to answer like human being to the questions based on the context provided. Please adhere to the following guidelines:

# 1. Use only the given context to answer the human input.
# 2. Do not use any additional knowledge or information.
# 3. Provide clear and concise answers in a maximum of four sentences.
# 4. If you don't know the answer, respond with "Sorry, I don't know "
# 5. If the human input is a wishing statement, respond appropriately.
# 6. If URL links are available in the context or metadata, include the most relevant link only once in your response, even if it appears multiple times. For example, if the context includes multiple references to https://nimbleproperty.net/, include this link only once in your answer.


# Context:
# {context}

# {chat_history}
# Human: {human_input}
# Chatbot:"""

#         prompt = PromptTemplate(
#             input_variables=["chat_history", "human_input", "context"],
#             template=template
#         )

#         q_chain = load_qa_chain(
#             llm,
#             chain_type="stuff",
#             memory=memory,
#             prompt=prompt
#         )

#         result = q_chain.invoke(
#             {"input_documents": reordering_docs, "human_input": question},
#             return_only_outputs=False
#         )
        
#         # user_specific_memory.add_user_message(question)
#         # user_specific_memory.add_ai_message(result["output_text"])
#         return result
#     except Exception as e:
#         logger.error(f"Error creating conversational chain: {e}")
#         return None
    


# class DataBaseLoader:
#     _instance = None
    
#     def __new__(cls, *args, **kwargs):
#         if cls._instance is None:
#             cls._instance = super(DataBaseLoader, cls).__new__(cls)
#             cls._instance._initialize(*args, **kwargs)
#         return cls._instance

#     def _initialize(self, embed_loc, text_loc, embedding_function):
#         #print("Initializing the database...")
#         self.vectors = Chroma(persist_directory=embed_loc, embedding_function=embedding_function)
#         self.document_text = retrieve_doc(text_loc)
        
#         bm25_retriever = BM25Retriever.from_documents(self.document_text)
#         bm25_retriever.k = 3
#         self.ensemble_retriever = EnsembleRetriever(
#             retrievers=[bm25_retriever, self.vectors.as_retriever()],
#             weights=[0.3, 0.7]
#         )
        
# def get_data_base_loader():
#     database_name = "kodefast"
#     loc = "DataBase-01/" + database_name
#     embed_loc = loc + "/Embeddings"
#     text_loc = loc + "/Texts"
#     embedding_function = embeddings 
#     return DataBaseLoader(embed_loc, text_loc, embedding_function)

# db_loader = get_data_base_loader()
# vectors = db_loader.vectors
# document_text = db_loader.document_text
# ensemble_retriever = db_loader.ensemble_retriever

# app6=FastAPI()
# class UserInput(BaseModel):
#     question: str
#     unique_id: str

# @app6.post("/")
# async def ask_question(userInput: UserInput):
#   try:
#       if vectors is None or document_text is None:
#           return JSONResponse(content={"error": "Resources not loaded properly"}, status_code=500)
      
#       result = get_conversational_chain( userInput.question, userInput.unique_id,vectors)
#       return JSONResponse(content={"answer": result["output_text"]}, status_code=200)
#   except Exception as e:
#       return JSONResponse(content={"error": str(e)}, status_code=500)


# # database_name=input("Enter the name of the Database:")
# # loc="DataBase/"+database_name
# # embed_loc=loc+"/Embeddings"
# # text_loc=loc+"/Texts"

# # print("hi")
# # print("------Choose the data Knowledge Source---\n")
# # choice=input("Enter 1 to get data from the url:\nEnter 2 to get the data from the local files:")
# # if choice=="1":
# #    url=input("Enter the url:")
# #    text=extract_text(url)
# #    print(text)
# #    chunks = get_text_chunks(text)
# #    store_doc(chunks,text_loc)
# #    vectors = get_vector_store2(chunks,embed_loc)
# # elif choice=="2":
# #    files=input("Enter the path of the files:")
# #    text=data_from_directory(files)
# #    chunks = get_text_chunks(text)
# #    store_doc(chunks, text_loc)g
# #    vectors = get_vector_store2(chunks, embed_loc)


# # # #loading the vectors and the text data from the local drive
# # vectors=Chroma(persist_directory=embed_loc,embedding_function=embeddings)
# # text=retrieve_doc(text_loc)
# # for i in range(10):
# #     question=input("Press Enter your question:")
# #     unique_id="user=01"
# #     result=get_conversational_chain(vectors,text,question,unique_id)
# #     print(result["output_text"])
# #     print("\n")



import os
import re
from operator import itemgetter
from typing import List, Tuple
from pymongo import MongoClient
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    format_document,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_transformers import LongContextReorder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import json
from langchain_google_genai import GoogleGenerativeAI
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.schema import Document
import time
from chatbot.add_user_data_to_vectordb import EmbeddingModelManager
from chatbot.logs import logger
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import Runnable
import asyncio
from chatbot.initilize_llm import initialize_llm_async
from chatbot.add_user_data_to_vectordb import MongoDBConnectionManager

GEMINI_API_KEY = os.getenv("GEMINI_PAID_KEY")

DEFAULT_MODEL_NAME = 'gemini-2.0-flash'
DEFAULT_API_KEY = GEMINI_API_KEY

api_key = GEMINI_API_KEY

#DEFAULT_API_KEY='AIzaSyAdSamToCULGAbZmvFC_G38vZ8fdfJSyjU'

#api_key = "AIzaSyAdSamToCULGAbZmvFC_G38vZ8fdfJSyjU"

#DEFAULT_API_KEY='AIzaSyAS4K2-y79iXICUTrj0q8OdQc848KClln0'

#api_key = "AIzaSyAS4K2-y79iXICUTrj0q8OdQc848KClln0"

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
google_llm = GoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key=api_key, temperature=0, verbose=True,timeout=600)

from langchain_openai import ChatOpenAI

openai_api_key = "sk-proj-4IhqqRg66dPO6sfiOJdtT3BlbkFJ80X197ya0nmyBOEMqpsG"
openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key, dimensions=768)
#llm = ChatOpenAI(model_name="gpt-4", api_key=openai_api_key,temperature=0,max_retries=2)
llm=google_llm
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embedding_manager = EmbeddingModelManager()
hf_embedding=embedding_manager.get_embedding_model()
hf_embedding=hf_embedding

reordering = LongContextReorder()

AI_CONNECTION_STRING = os.getenv("AI_CONNECTION_STRING") 
DB_NAME = "chatbot"
COLLECTION_NAME = "user_files"
VECTOR_SEARCH_INDEX = "vector_index"

# from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# model_name = "BAAI/bge-base-en-v1.5"
# model_kwargs = {"device": "cpu"}
# encode_kwargs = {"normalize_embeddings": True}
# hf_embedding = HuggingFaceBgeEmbeddings(
#     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
# )

class DataBaseLoader:
    """
    Singleton class to load and manage the connection with MongoDB.
    
    Attributes:
        client (MongoClient): The MongoDB client.
        db (Database): The MongoDB database.
        collection (Collection): The collection within the database.
        embeddings: Embeddings model used for vector search.
        vector_search (MongoDBAtlasVectorSearch): The vector search instance.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DataBaseLoader, cls).__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance


    def _initialize(self):
        logger.info("Initializing MongoDB connection and vector search.")
        self.client = MongoClient(AI_CONNECTION_STRING)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.embeddings =hf_embedding

        self.vector_search = MongoDBAtlasVectorSearch.from_connection_string(
            AI_CONNECTION_STRING,
            f"{DB_NAME}.{COLLECTION_NAME}",
            self.embeddings,
            index_name=VECTOR_SEARCH_INDEX
        )
        logger.info("MongoDB connection and vector search initialized.")
        #print("     connecing the mongodb          \n")

    
    def get_documents(self, admin_id):
        """
        Retrieves documents from the MongoDB collection for a specific admin ID.
        
        Args:
            admin_id (str): The admin ID to filter documents.

        Returns:
            List[Document]: A list of documents for the given admin ID.
        """
        logger.info(f"Fetching documents for admin_id: {admin_id}")
        query = {"admin_id": admin_id}
        projection = {"text": 1}
        results = self.collection.find(query, projection)
        logger.info(f"Fetched documents for admin_id: {admin_id}")
        return [Document(page_content=doc["text"]) for doc in results]
    
    
    def get_ensemble_retriever(self, admin_id,llm_model):
        """
        Creates and returns an ensemble retriever for the given admin ID.
        
        Args:
            admin_id (str): The admin ID for which the retriever is created.

        Returns:
            EnsembleRetriever: An ensemble retriever instance.
        """
        #start_time=time.time()
        logger.info(f"Creating ensemble retriever for admin_id: {admin_id}")
        documents = self.get_documents(admin_id)
        
        retriever = self.vector_search.as_retriever(
        search_type = "similarity_score_threshold",
        search_kwargs = {
            "k": 10,
            "score_threshold": 0.25,
            "pre_filter": { "admin_id": { "$eq": admin_id } }
        })
        retriever_from_llm=MultiQueryRetriever.from_llm(retriever=retriever,llm=llm_model)

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 10
        #print(time.time()-start_time)
        logger.info("Ensemble retriever created.")
        return [EnsembleRetriever(retrievers=[bm25_retriever, retriever_from_llm],weights=[0.3, 0.7]),retriever]
    

def get_data_base_loader():
    return DataBaseLoader()

db_loader = get_data_base_loader()

async def get_answer(question, ensemble_retriever, chat_history,llm_model,support_email): 
    print("support_email",support_email)  
    st1=time.time() 
    _template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.
    Chat History:
    {chat_history}
    Follow-Up Input: {question}
    Standalone question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

     # RAG answer synthesis prompt
    template = """====== General Settings ======

### Bot Persona
You are an AI Chatbot representative for both sales and customer support.
Answer user questions and lead the conversation to provide users with the best visitor experience.
Respond with a professional, friendly, and supportive tone.
##summary of the docuement Instructions:
identify the docuement use case , key sections and main themes of the document.Present them in a clear hierarchy with headings with clear description and bullet points.
- keep the heading like "Here’s a structured analysis of "file name"s profile from the uploaded document:
### Audience
Your users are seeking assistance with product information, purchases,information in the csv/excel and general inquiries about the store.
 
### Internal Thinking
<think>
Reflect on the user's query and context.
Think step by step
Formulate a concise, accurate, and user-friendly response.
If a source or website is available, always attach the relevant URL to the answer, even if the user didn’t request it.
please Do not display internal thoughts to users
</think>
 
===== Formatting Guideline ======
### Formatting Guidance
URLs must be formatted in markdown link syntax: `(http://example.com)`.
Email addresses should be in bold, like this: **example@gamil.com**.
 
===== Answering Guidelines =====
### Guidelines
**Information Collection**:
Ask for relevant user information to personalize responses.
Confirm whether your response was satisfactory at the end of the conversation.
If the user's input is a greeting or well-wishing statement, respond appropriately with emojis

**General Answers**:
Only include specifics about products that are in the knowledge base. Do not answer with your  assumption.
Keep answers concise and under 30 words strictly. If an answer exceeds 30 words, paraphrase to reduce length.

**Product Search Questions**:
List no more than three items in responses.
Include specifics about products that are in the knowledge base without assuming additional sizes, colors, etc.
Do not choose arbitrary units for pricing, weight, etc.
- Automatically include a reference link to a product page, documentation, or article — even if not asked.

**Product Recommendations and Website Information**:
- Share each as a bullet point with a markdown-formatted link: `[Product Name](http://product-url.com)`
Ensure the link is relevant to the context retrieved or web data available.

**'Talk to Business Owner or Human' Intent**:
Always include the store email address in the response.

**Unknown Questions**:
Respond that you are unable to answer based on your knowledge base.
Include the store email address in the response please look over the avaliable information in the knowledge about the  email and give in response
 
===== Examples =====
### Examples
**Example 1: Informational Query**
**User**: What is LVN (Licensed Vocational Nurse)?
**Bot**: A Licensed Vocational Nurse (LVN) provides basic nursing care under the supervision of RNs and doctors. 🏥  
[Learn more](https://www.nursinglicensure.org/articles/lvn/)

**Example 2: Product Inquiry**
**User**: what is the cost of esigns? with source url
**Bot**: eSigns offers flexible pricing plans to suit your needs. 🤑 You can choose from the Individual Plan, Business Essential, Business Professional, or Business Enterprises. 💼  
The Individual Plan costs $9.99/month, Business Essential is $19.99/month, and Business Professional is $29.99/month. 💰  
[View more](https://esigns.io/pricing/)

**Example 3: Seeking Human Assistance**
**User**: I need to talk to the store owner.
**Bot**: Please contact the store owner at **give {support_email} by seeing the provided knowledgebase**. Is there anything else I can assist you with?
 
**Example 4: Unknown Question**
**User**: What is the weather like today?
**Bot**: I'm unable to answer that question. For further assistance, please contact us at '{{support_email}}'.

 
Here is the context you will use to answer questions:
 
<context>
{context}
</context>"""



    ANSWER_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
        ]
    )

    # Conversational Retrieval Chain
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

    def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
        # Combine documents into a single context
        #print(docs)
        formatted_docs = [format_document(doc, document_prompt) for doc in docs]
        #print(formatted_docs)
        return document_separator.join(formatted_docs)

    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    # User input
    class ChatHistory(BaseModel):
        chat_history: List[Tuple[str, str]] = Field(..., description="The chat history as a list of tuples")
        question: str

    _search_query = RunnableBranch(
        # If input includes chat_history, we condense it with the follow-up question
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ), 
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            |
            CONDENSE_QUESTION_PROMPT
            | llm_model
            | StrOutputParser(),
        ),
        # Else, we have no chat history, so just pass through the question
        RunnableLambda(itemgetter("question")),
    )

    _inputs = RunnableParallel(
        {
            "question": lambda x: x["question"],
            "support_email": lambda x: x["support_email"],
            "chat_history": lambda x: _format_chat_history(x["chat_history"]),
            "context": _search_query | ensemble_retriever | _combine_documents,
        }
    ).with_types(input_type=ChatHistory)

    chain = _inputs | ANSWER_PROMPT | llm_model | StrOutputParser()

    response = chain.invoke({
                "question": question,
                "support_email":support_email,
                "chat_history": chat_history
            })
    #print("answer function time is",time.time()-st1)
    
    return response





class ChatHistoryManager:
    """
    Manages chat history stored in MongoDB.

    Attributes:
        client (MongoClient): The MongoDB client.
        db (Database): The MongoDB database.
        collection (Collection): The collection within the database.
    """
    def __init__(self, uri, db_name, collection_name):
        """
        Initializes the MongoDB connection and selects the collection.

        Args:
            uri (str): MongoDB connection URI.
            db_name (str): Name of the database.
            collection_name (str): Name of the collection.
        """
        #print("     connecing the mongodb          \n")
        logger.info("Initializing ChatHistoryManager with MongoDB connection.")
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        logger.info("ChatHistoryManager initialized.")

    def store_message(self, session_id, question, answer,admin_id):
        """
        Stores a message in the chat history for a specific session.

        Args:
            session_id (str): The session ID.
            question (str): The user's question.
            answer (str): The bot's answer.
            admin_id (str): The admin ID.
        """
        #logger.info(f"Storing message for session_id: {session_id}, admin_id: {admin_id}")
        logger.info(f"Storing message for session_id: {session_id}, admin_id: {admin_id}")
        try:
            self.collection.update_one(
                {'session_id': session_id},
                {
                    '$set': {"admin_id": admin_id},
                    '$push': {
                        'chat_history': {
                            '$each': [{'timestamp': datetime.utcnow(), 'question': question, 'answer': answer}],
                            '$slice': -20 
                        }
                    }
                },
                upsert=True
            )
            #logger.info(f"Message stored for session_id: {session_id}, admin_id: {admin_id}")
        except Exception as e:
            logger.error(f"Error storing message for session_id: {session_id}, admin_id: {admin_id} - {str(e)}")

    def get_last_10_messages(self, session_id):
        """
        Retrieves the last few meaningful question-answer pairs from the chat history for a specific session.
    Excludes non-informative messages like greetings, thanks, and similar common phrases.

        Args:
            session_id (str): The session ID.

        Returns:
            List[Tuple[str, str]]: A list of the last Few question-answer pairs.
        """


        try:
            session = self.collection.find_one({'session_id': session_id}, {'_id': 0, 'chat_history': 1})
            if session and 'chat_history' in session:
                return [(entry['question'], entry['answer']) for entry in session['chat_history']]
            return []
        except Exception as e:
            logger.error(f"Error fetching chat history for session_id: {session_id} - {str(e)}")

    
AI_CONNECTION_STRING = os.getenv("AI_CONNECTION_STRING") 
DB_NAME = "AI-ML"
COLLECTION_NAME = "chatbot-History"
chat_manager = ChatHistoryManager(AI_CONNECTION_STRING, DB_NAME, COLLECTION_NAME)



def remove_think_step(text):
    thinking_pattern=r"\n?<think>.*?</think>\n?"
    clean_text=re.sub(thinking_pattern,"",text,flags=re.DOTALL)
    return clean_text





class GenerateQuestionsOutput(BaseModel):
    questions: List[str] = Field(description="List of suggested questions")

parser = JsonOutputParser(pydantic_object=GenerateQuestionsOutput)
response_parser = parser.get_format_instructions()
prompt_template = PromptTemplate(
    template="""
====== General Settings ======

### Bot Persona
You are an expert at generating very concise follow-up questions that reflect how a **user** might continue their conversation with the chatbot based on the current question, context, and previous chat history. Output the questions in JSON format using the provided instructions: {format_instructions}. Keep the responses short and user-like.

### Audience
Users are seeking brief, natural-sounding questions they might ask to further their inquiry. Use simple language, and ensure the questions are directly related to the context provided and current.

### Internal Thinking
<think>
Think like you're the user chatting with the chatbot. What follow-up questions might you naturally ask the chatbot next, given the current question, context, and previous chat history?
Ensure each follow-up question is short, under 50 tokens, and written in simple, user-like language.
Only ask questions that directly relate to the current conversation.
If the user asks a general greeting (e.g., "hi", "how are you", "hello") at the start, generate follow-up questions or suggestions related to the product/service being discussed.
Do not display internal thoughts to users.
</think>

===== Formatting Guideline ======
### Formatting Guidance
Questions should be separated by a newline.
Maintain a conversational and natural tone, as if the user is continuing their inquiry.
For general greetings, ensure follow-up questions are relevant to the product or service.

===== Follow-Up Question Generation Guidelines =====
### Guidelines
Generate **only minimum 1 and maximum 3 follow-up questions** based on the user's current question, context, and previous chat history.
- **Strict Relevance**: Generate **only** follow-up questions that can be answered based on the **context**, **current question**, and **chat history**. Do not include any questions beyond the provided information.
- **Clarity**: Formulate concise questions that reflect a user's natural speaking style.
- **Engagement**: Encourage further interaction by predicting what the user might ask next.

===== Examples =====

### Example 1: Based on a General Greeting
*Current Question*:
"Hi", "Hello", "Hey", "Hi there", "Greetings", "Hey there",
"Howdy", "What's up?", "Yo!", "Good day", "Hey, how's it going?",
"Hiya", "Hello there", "Sup?", "Hey, what's up?", "Hi, nice to see you!",
"Hey, how have you been?", "Hi, how are you?", "Good to see you!",
"Hello, friend!", "Hey buddy!", "What’s happening?", "Hello, how are things?",
"Hey, long time no see!", "How's everything?", "Hi, what’s new?",
"Hey, what’s going on?", "How’s life?", "Greetings, friend!", "Hi, howdy!"
*Generated User-Like Follow-Up Questions*:

  "answer": "👋  How can I help you today? 😊",
  "follow-up questions": 
    "What's the best way to use this software?",
    "Can you tell me more about the features?",
    "What are some of the benefits of using this software?"
  
### Example 2: Based on Current Question
*Current Question*: "What are the features of the premium plan?"
*Generated User-Like Follow-Up Questions*:
1. "Can you explain how the premium features work?"
2. "Which features stand out the most?"
3. "Would the premium plan suit my needs?"

### Example 3: Based on Context
*Context*: The user is interested in pricing and benefits of plans.
*Generated User-Like Follow-Up Questions*:
1. "What’s the best plan for saving money?"
2. "How do the benefits compare between plans?"
3. "Which plan gives the most value?"

### Example 4: Based on Previous Chat History
*Previous Chat History*: The user has been asking about various plans and their benefits.
*Generated User-Like Follow-Up Questions*:
1. "Which plan do you think is better based on what we discussed?"
2. "Can you help me pick the best one?"
3. "Would you recommend the premium option?"

Here is the context you will use to generate suggestive questions:

<context>
{context}
</context>

Here is the current question:
<question>
{current_question}
</question>

Here is the previous chat history:
<previous_chat_history>
{chat_history}
</previous_chat_history>
""",


    input_variables=["chat_history", "current_question", "context",],
    partial_variables={"format_instructions":response_parser}

)



async def generate_follow_up_questions(chat_history: List[Tuple], current_question: str,ensemble_retreiver,llm_model) -> any:
    """
    Generates five follow-up questions based on the chat history, current question, and context.

    Parameters:
    chat_history (str): The history of the chat.
    current_question (str): The current question being asked.
    context (str): Additional context relevant to the current question.

    Returns:
    List[str]: The generated follow-up questions.
    """
    # Run the LLM chain with the input data
    
    st=time.time()    
    context=ensemble_retreiver.invoke(current_question)
    template_chain: Runnable = prompt_template | llm_model | parser
    formatted_chat_history = ''.join([f'<Question>{question} <Answer>{answer}\n ' for question, answer in chat_history])
    

    response = template_chain.invoke({
        "chat_history": formatted_chat_history,
        "current_question": current_question,
        "context": context
    })
    #print("followup time is",time.time()-st)
    return response


app6= FastAPI()
from chatbot.get_chat_histroy import client
db=client["chatbot"]
config_collection=db["LLM_configuration"]
support_collection=db["user_files"]

class ChatHistoryResponse(BaseModel):
    answer: str
    questions:list
class LLMConfigRequest(BaseModel):
    Model_name:str=Field(...,min_length=1,description="The model name should be Non-Empty String")
    api_key: str = Field(..., min_length=1, description="The API key must be a non-empty string")
    admin_id: str = Field(..., min_length=1, description="The admin ID must be a non-empty string")
class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question must be a non-empty string")
    unique_id: str = Field(..., min_length=1, description="The unique ID must be a non-empty string")
    admin_id: str = Field(..., min_length=1, description="The admin ID must be a non-empty string")


async def save_llm_config(admin_id: str, model_name: str, api_key: str):
    """
    Stores or updates the LLM configuration for an admin in MongoDB.

    Args:
        admin_id (str): Admin ID.
        model_name (str): LLM model name.
        api_key (str): LLM API key.

    Returns:
        None
    """
    try:
        llm_config = {
            "admin_id": admin_id,
            "model_name": model_name,
            "api_key": api_key
        }
        logger.info(f"Attempting to save LLM configuration for admin_id: {admin_id}")
        await asyncio.to_thread(config_collection.update_one, {"admin_id": admin_id}, {"$set": llm_config}, upsert=True)
        logger.info(f"LLM configuration saved for admin_id: {admin_id}")
    except Exception as e:
        logger.error(f"Error saving LLM configuration for admin_id: {admin_id} - {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error saving LLM configuration: {str(e)}")


async def get_llm_config(admin_id: str):
    """
    Retrieves the LLM configuration for an admin from MongoDB.

    Args:
        admin_id (str): Admin ID.

    Returns:
        dict: The LLM configuration for the admin.
    """
    try:
        logger.info(f"Retrieving LLM configuration for admin_id: {admin_id}")
        llm_config = await asyncio.to_thread(config_collection.find_one, {"admin_id": admin_id})
        if not llm_config:
            logger.warning(f"LLM configuration not found for admin_id: {admin_id}. Using default configuration.")
            return {"model_name": DEFAULT_MODEL_NAME, "api_key": DEFAULT_API_KEY}
        logger.info(f"LLM configuration retrieved successfully for admin_id: {admin_id}")
        return {"model_name": llm_config.get("model_name"), "api_key": llm_config.get("api_key")}
    except Exception as e:
        logger.error(f"Error retrieving LLM configuration for admin_id: {admin_id} - {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving LLM configuration: {str(e)}")


async def get_support_email(admin_id: str):
    """
    Retrieves the support email configuration for an admin from MongoDB.
 
    Args:
        admin_id (str): Admin ID.
 
    Returns:
        dict: The support email configuration for the admin.
    """
    try:
        logger.info(f"Retrieving support email for admin_id: {admin_id}")
       
        # Use to_thread to run blocking I/O in a non-blocking way
        support_config = await asyncio.to_thread(support_collection.find_one, {"admin_id": admin_id})
       
        if not support_config:
            logger.warning(f"Support email not found for admin_id: {admin_id}. Using default support email.")
            return {"support_email":""}
       
        # Log the configuration details for debugging, avoiding sensitive data
        logger.info(f"Support configuration retrieved for admin_id: {admin_id}, Title: {support_config.get('title')}, Language: {support_config.get('language')}")
       
        # Extract the support_email, or use default if missing
        support_email = support_config.get("support_email", "")
 
        # Return the relevant support email
        return support_email
 
    except Exception as e:
        # Log the error with stack trace for debugging
        logger.error(f"Error retrieving support email for admin_id: {admin_id} - {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving support email: {str(e)}")
 


# @app6.post("/", response_model=ChatHistoryResponse)
# async def chat(request: ChatRequest):
#     """
#     Handles the chat requests and returns the chatbot's response.

#     Args:
#         request (ChatRequest): The incoming request containing the question, unique ID, and admin ID.

#     Returns:
#         ChatHistoryResponse: The chatbot's response.
#     """
#     try:
#         llm_config = await get_llm_config(request.admin_id)
#         support_email=await get_support_email(request.admin_id)
#         model_name = llm_config["model_name"]
#         api_key = llm_config["api_key"]
#         llm_model=initialize_llm_async(model_name=model_name,api_key=api_key)

#         chat_history = chat_manager.get_last_10_messages(request.unique_id)
#         #print("the chat history is \n:,",chat_history)
#         retrievers = db_loader.get_ensemble_retriever(request.admin_id,llm_model)
#         #context=ensemble_retriever.invoke(request.question)
#         #follow_up_questions = generate_follow_up_questions(chat_history, request.question,ensemble_retriever)
#         #print("######################/n",follow_up_questions)

#         # initial_response = get_response(request.question)

#         # if initial_response.lower() == "demo":
#         #     response =initial_response
#         #     chat_manager.store_message(request.unique_id, request.question, "Demo Scheduled successfully",request.admin_id)
#         #     return ChatHistoryResponse(answer=response,questions=[])
#         # else:            
#         #follow_up_questions_task = asyncio.create_task(generate_follow_up_questions(chat_history, request.question, ensemble_retriever))
#         #answer_task = asyncio.create_task(get_answer(request.question, ensemble_retriever, chat_history))
#         follow_up_questions, response = await asyncio.gather(generate_follow_up_questions(chat_history, request.question, retrievers[1],llm_model),get_answer(request.question, retrievers[0], chat_history,llm_model,support_email))
#         if not response.strip():
#                 response = "Could you Please rephrase the question with more context?"
#         if response.startswith("AI:"):
#                 response = response[len("AI:"):].strip()

#         response=remove_think_step(response)
#         chat_manager.store_message(request.unique_id, request.question, response,request.admin_id)
        
#         return ChatHistoryResponse(answer=response,questions=follow_up_questions["questions"])



#     except Exception as e:
#         logger.error(f"Error handling chat request: {str(e)}",exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

@app6.post("/", response_model=ChatHistoryResponse)
async def chat(request: ChatRequest):
    """
    Handles the chat requests and returns the chatbot's response.

    Args:
        request (ChatRequest): The incoming request containing the question, unique ID, and admin ID.

    Returns:
        ChatHistoryResponse: The chatbot's response.
    """
    try:
        logger.info(f"Processing chat request for unique_id: {request.unique_id}")

        # Load LLM configuration
        llm_config = await get_llm_config(request.admin_id)
        support_email = await get_support_email(request.admin_id)
        model_name = llm_config["model_name"]
        api_key = llm_config["api_key"]
        llm_model = initialize_llm_async(model_name=model_name, api_key=api_key)

        # Open MongoDB connection
        mongo_manager = MongoDBConnectionManager()
        collection = mongo_manager.get_collection()
        logger.info(f"MongoDB connection established for admin_id: {request.admin_id}")

        # Retrieve chat history
        chat_history = chat_manager.get_last_10_messages(request.unique_id)
        logger.info(f"Chat history retrieved for unique_id: {request.unique_id}")

        # Load retrievers
        retrievers = db_loader.get_ensemble_retriever(request.admin_id, llm_model)
        logger.info(f"Retrievers initialized for admin_id: {request.admin_id}")

        # Generate follow-up questions and chatbot response asynchronously
        follow_up_questions, response = await asyncio.gather(
            generate_follow_up_questions(chat_history, request.question, retrievers[1], llm_model),
            get_answer(request.question, retrievers[0], chat_history, llm_model, support_email)
        )

        # Validate response
        if not response.strip():
            response = "Could you please rephrase the question with more context?"
        if response.startswith("AI:"):
            response = response[len("AI:"):].strip()

        # Remove unnecessary AI steps
        response = remove_think_step(response)

        # Store chat response
        chat_manager.store_message(request.unique_id, request.question, response, request.admin_id)
        logger.info(f"Chat response stored for unique_id: {request.unique_id}")

        return ChatHistoryResponse(answer=response, questions=follow_up_questions["questions"])

    except Exception as e:
        logger.error(f"Error handling chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

    finally:
        # Close MongoDB connection after request is completed
        mongo_manager.close_connection()
        logger.info(f"MongoDB connection closed for admin_id: {request.admin_id}")

@app6.post("/configure-llm")   
async def configure_llm(request: LLMConfigRequest):
    """
    Stores the LLM configuration for a specific admin. 
    The configuration will be used for all subsequent requests made by the admin.

    Args:
        request (LLMConfigRequest): Contains model name, API key, and admin ID.

    Returns:
        dict: Confirmation of the successful configuration.
    """
    logger.info(f"Received request to configure LLM for admin_id: {request.admin_id}")
    await save_llm_config(request.admin_id, request.Model_name, request.api_key)
    logger.info(f"LLM configuration updated successfully for admin_id: {request.admin_id}")
    return {"message": "LLM configuration updated successfully"}