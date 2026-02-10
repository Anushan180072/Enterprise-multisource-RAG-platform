import os
from urllib.parse import urlparse
from typing import List
from langchain.schema import Document
from chatbot.Image_extract import get_image_data_from_url
from chatbot.files_extract import get_file_data_from_url 
from chatbot.logs import logger

VALID_IMAGE_EXTENSIONS = [
    'jpeg', 'png', 'gif', 'bmp', 
    'tiff', 'webp', 'ico', 'svg'
]

VALID_FILE_EXTENSIONS = ['pdf', 'txt', 'csv', 'pptx', 'docx', 'xlsx', 'json']


def get_file_type_from_url(url: str) -> str:
    try:
        parsed_url = urlparse(url)
        file_name = os.path.basename(parsed_url.path)
        file_extension = file_name.split(".")[-1].lower()
        logger.debug(f"Extracted file extension '{file_extension}' from URL '{url}'")
        return file_extension
    
    except Exception as e:
        logger.error(f"Error extracting file type from {url}: {str(e)}")
        return None


def process_s3_urls(urls: List[str]) -> List[Document]:
    """
    Process a list of S3 URLs to identify file types and call appropriate functions to process them.
    
    Args:
        urls (List[str]): List of S3 URLs to process.
        
    Returns:
        List[Document]: A list of LangChain documents.
    """
    documents = []
    
    for url in urls:
        logger.info(f"Processing URL: {url}")
        try:
            file_type = get_file_type_from_url(url)
            
            if file_type in VALID_IMAGE_EXTENSIONS:
                logger.info(f"File type '{file_type}' recognized as an image")
                image_data = get_image_data_from_url(url)
                if image_data:
                    documents.extend([image_data])
                    logger.debug("Image data extracted from")

            elif file_type in VALID_FILE_EXTENSIONS:
                logger.info(f"File type '{file_type}' recognized as a document")
                file_data = get_file_data_from_url(url)
                if file_data:
                    documents.extend(file_data)
                    logger.debug(f"File data extracted from ")
            else:
                logger.info(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
    
    return documents


