import requests
import json
import os
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from tempfile import NamedTemporaryFile
import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredExcelLoader, UnstructuredPowerPointLoader, Docx2txtLoader
from langchain.schema import Document
from chatbot.logs import logger

def fetch_file_from_url(url):
    """
    Fetches the content of a file from a given URL.

    Args:
        url (str): The URL of the file to fetch.

    Returns:
        bytes: The content of the file if the request is successful, None otherwise.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        logger.info("S3 File fetched successfully")
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error  fetching s3 file: {e}")
        return None

def get_data_from_file(file_path, file_type, file_name):
    """
    Processes a file to extract text content and convert it into a list of LangChain Document objects.

    Args:
        file_path (str): The path to the file.
        file_type (str): The type of the file (e.g., 'pdf', 'txt', 'csv', 'pptx', 'docx', 'xlsx', 'json').
        file_name (str): The name of the file.

    Returns:
        list of Document: A list of LangChain Document objects containing the text content and metadata.
    """
    logger.info(f"Processing file: {file_path}, type: {file_type}, name: {file_name}")
    try:
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()

        elif file_type in ['txt', 'text']:
            loader = TextLoader(file_path)
            documents = loader.load()

        elif file_type == 'csv':
            loader = CSVLoader(file_path)
            documents = loader.load()

        elif file_type == 'pptx':
            loader = UnstructuredPowerPointLoader(file_path)
            documents = loader.load()

        elif file_type == 'docx':
            loader = Docx2txtLoader(file_path)
            documents = loader.load()

        elif file_type == 'xlsx':
            loader = UnstructuredExcelLoader(file_path, mode="elements")
            xlsx_data = loader.load()
            html_content = str(xlsx_data)
            soup = BeautifulSoup(html_content, 'html.parser')
            rows = soup.find('tbody').find_all('tr')
            excel_data = []
            for row in rows:
                cells = row.find_all('td')
                excel_data.extend([cell.get_text() for cell in cells])
            text_content = '\n'.join(excel_data)
            documents = [Document(page_content=text_content, metadata={"source": file_name})]

        elif file_type == 'json':
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            text_content = json.dumps(data, indent=2)
            documents = [Document(page_content=text_content, metadata={"source": file_name})]

        else:
            logger.error(f"Unsupported file type: {file_type}")
            return []

        # Add metadata to documents
        for doc in documents:
            doc.metadata["source"] = file_name
        
        #print(documents)
        logger.info(f"Documents processed successfully for file: {file_name}")
        return documents

    except Exception as e:
        logger.error(f"Error processing file {file_name}: {e}")
        return []
    

def get_file_data_from_url(url):
    """
    Downloads a file from a given URL, processes it based on its type, and returns the file's data in LangChain Document format.

    Args:
        url (str): The URL of the file to download.

    Returns:
        list of Document: A list of LangChain Document objects containing the text content and metadata.
    """
    parsed_url = urlparse(url)
    file_name = os.path.basename(parsed_url.path)
    file_type = file_name.split(".")[-1].lower()

    file_content = fetch_file_from_url(url)
    if file_content is None:
        return []

    temp_file_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            temp_file_path = temp_file.name
            #print(temp_file_path)
            documents = get_data_from_file(temp_file_path, file_type, file_name)
            logger.info(f"Documents retrieved from file: {file_name}")
            #print(documents)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Temporary file path removed: {temp_file_path}")
        
    return documents

