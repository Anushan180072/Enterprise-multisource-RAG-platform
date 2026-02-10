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
# api_key = "AIzaSyD2xh0ykHxlAPLw7TTM5ECe5r3XlpwkGXs"
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
# from langchain_openai import OpenAIEmbeddings
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
# AI_CONNECTION_STRING = os.getenv("AI_CONNECTION_STRING") 
# db_name = "AI-ML"
# collection_name = "chatbot_chat_history"

# def get_user_memory(user_id):
#     return MongoDBChatMessageHistory(
#         connection_string=AI_CONNECTION_STRING,
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
#         template ="""I want you to act as the Nimble assistant chatbot and your name is Mike.  Your job is to answer like a human being to the questions with suitable 3-D Emojis and Icons based on the context provided and Avoid mentioning AI at starting of answer and Try to give maximum Emojis . Please adhere to the following guidelines as it is very important to my Life, If you follow Them, I will give you $4000 bucks.
 
# 1. If the user's question involves pricing, cost, or subscription information of nimble property, and the context does not provide enough information, provide the following contact details in a well-structured manner: "You can contact  the following to know more about our sales:
#    - 📧: info@nimbleproperty.net
#    - 📞: +1 (866)-964-6253
#    Also, Here is the  link to schedule 📅 Demo with us: "https://nimbleproperty.net/demo". Avoid saying "I don't know."
# 2. Use only the given context to answer the user's input.
# 3. Do not use any additional knowledge or information.
# 4. Provide clear and concise answers, limited to a maximum of Three sentences.
# 5. If you don't know the answer, respond with, "Sorry, I don't know. Could you rephrase the question 🔄!
# 6. If the user's input is a greeting or well-wishing statement, respond appropriately.
# 7. If URL links are available in the context, include the most relevant link to question only once in your response, even if it appears multiple times. For example, if the context includes multiple references to "https://nimbleproperty.net/demo", include this link only once in your answer like here is the reference link https://nimbleproperty.net/demo
 
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
#     database_name = "Nimble_property"
#     loc = "DataBase-01/" + database_name
#     embed_loc = loc + "/Embeddings"
#     text_loc = loc + "/Texts"
#     embedding_function = embeddings 
#     return DataBaseLoader(embed_loc, text_loc, embedding_function)

# db_loader = get_data_base_loader()
# vectors = db_loader.vectors
# document_text = db_loader.document_text
# ensemble_retriever = db_loader.ensemble_retriever



# prompt_template_demo = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant designed to understand user intents. The user is interacting with a chatbot that can schedule demos. Classify the following question as either 'yes' if the user wants to schedule a demo, or 'no' if the user does not have this intention. Respond with only the classification."),
#     ("human", "{question}")
# ])

# def classify_intent(question):
#     prompt = prompt_template_demo.format(question=question)
#     response = llm.invoke(prompt)
#     return response.strip()

# def get_response(user_question):
#     intent = classify_intent(user_question)
#     if intent.lower() == "yes":
#         return "demo"
#     else:
#         return "not_interested"

# app7=FastAPI()
# class UserInput(BaseModel):
#     question: str
#     unique_id: str


# @app7.post("/")
# async def ask_question(userInput: UserInput):
#     try:
#         if vectors is None or document_text is None:
#             return JSONResponse(content={"error": "Resources not loaded properly"}, status_code=500)
#         intent = classify_intent(userInput.question)

#         if intent.lower() == "yes":
    
#             return JSONResponse(content={"answer": "demo"}, status_code=200)
#         else:
            
#             result = get_conversational_chain(userInput.question, userInput.unique_id, vectors)
#             output_text = result.get("output_text", "")

#             if not output_text.strip():
#                 return JSONResponse(content={"answer": "Could you please rephrase the question with more context"}, status_code=200)

#             return JSONResponse(content={"answer": output_text}, status_code=200)

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)



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
