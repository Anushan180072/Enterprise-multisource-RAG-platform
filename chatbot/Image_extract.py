import google.generativeai as genai
import mimetypes
from PIL import Image
import io
import requests
import tempfile
import logging
from urllib.parse import urlparse
from langchain.schema import Document
from dotenv import load_dotenv
import os
from chatbot.logs import logger
load_dotenv()
api_key = os.getenv("google_api_key")
genai.configure(api_key=api_key)
MODEL_CONFIG = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=MODEL_CONFIG, safety_settings=safety_settings)
VALID_FORMATS = [
    'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 
    'image/tiff', 'image/webp', 'image/x-icon', 'image/svg+xml'
]

def get_file_details_from_url(url):
    """Extracts file details such as filename and file type from a URL."""
    logger.info(f"Extracting document file details from S3URL: {url}")
    parsed_url = urlparse(url)
    file_name = parsed_url.path.split('/')[-1]
    file_type = file_name.split('.')[-1].lower()
    logger.info(f"Extracted file name: {file_name}, file type: {file_type} from the document file")
    return file_name, file_type

def download_image(s3_url):
    """Downloads an image from the given S3 URL and checks its MIME type."""
    logger.info(f"Starting image download from s3URL: {s3_url}")
    try:
        response = requests.get(s3_url)
        response.raise_for_status()
        file_bytes = response.content
        mime_type = response.headers.get('Content-Type')
        if not mime_type or mime_type not in VALID_FORMATS:
            raise ValueError(f"Invalid image format: {mime_type}")
        return file_bytes, mime_type
    except requests.RequestException as e:
        logger.error(f"Error downloading image from S3URL {s3_url}: {e}")
        raise
    except ValueError as e:
        logger.error(e)
        raise

def get_image_data_from_url(s3_url):
    """Generates text from an image at the given S3 URL using the generative model."""
    logger.info(f"Processing image data from URL: {s3_url}")
    try:
        system_prompt = """
               You are an expert at extracting text from various types of images. Given an image, analyze it step by step and extract as much text as possible. If you cannot extract the text, provide a detailed explanation of the image content and characteristics in a manner that is useful for performing Q&A based on that explanation.
               """
        file_bytes, mime_type = download_image(s3_url)

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{mime_type.split('/')[1]}") as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name


        with Image.open(temp_file_path) as img:
            input_prompt = [system_prompt, img]
            #logger.info(f"MIME Type: {mime_type}")
            #logger.info(f"Input Prompt: {input_prompt}")
            
            response = model.generate_content(
                input_prompt,
                safety_settings=safety_settings,
                generation_config=genai.types.GenerationConfig(temperature=0)
            )
            
            #logger.info(f"API Response: {response}")

            if response and hasattr(response, 'parts'):
                parts = response.parts
                content = []
                for part in parts:
                    if hasattr(part, 'text'):
                        content.append(part.text)
                    else:
                        content.append(str(part)) 
                document_text = " ".join(content)
                file_name, _ = get_file_details_from_url(s3_url)
                metadata = {"filename": file_name}
                logger.info(f"Successfully extracted data from image {file_name}")
                return Document(page_content=document_text, metadata=metadata)
            else:
                logger.warning("No 'parts' attribute found in response.")
                return None
        
    except Exception as e:
        logger.error(f"Error generating/extracting content from image: {e}")
        raise
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)