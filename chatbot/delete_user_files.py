from chatbot.add_user_data_to_vectordb import classify_urls,data_from_youtube_web,atlas_collection
from chatbot.Get_data_from_Youtube import get_channel_id,get_channel_videos_links
from chatbot.get_all_subpage_links import get_all_sitemap_urls
from pymongo import MongoClient, results
from typing import List, Optional
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from chatbot.logs import logger
from chatbot.validate_web_url import is_root_page

def delete_documents(source: List[str], admin_id: str) -> str:
    """
    Delete documents from a MongoDB collection based on a list of URLs (source)
    and the bot ID (admin_id).

    Args:
        source (List[str]): A list of file names or other knowledge (e.g., weblinks, YouTube links).
        admin_id (str): The bot ID to match in the documents.

    Returns:
        str: A success message with the number of documents deleted, or an error message if the operation fails.
    """
    logger.debug("Starting delete_documents function")
    try:
        filter_documents = {"source": {"$in": source}, "admin_id": admin_id}
        result = atlas_collection.delete_many(filter_documents)
        logger.info(f"Deleted documents count: {result.deleted_count}")
        if result.deleted_count > 0:
            return True
        else:
            return False

    except Exception as e:
        return f"An error occurred: {e}"
    
    

def extract_youtube_video_ids(url_list: List[str]) -> List[str]:
    """
    Extracts video IDs from a list of YouTube URLs (long, short, or Shorts formats).

    Args
        url_list: List[str]: List of YouTube URLs as strings. (e.g., ["https://www.youtube.com/watch?v=VIDEO_ID", "https://youtu.be/VIDEO_ID"])
    
    Returns
        List[str]: List of extracted YouTube video IDs as strings. (e.g., ["VIDEO_ID1", "VIDEO_ID2"])
    """
    logger.debug("Starting extract_youtube_video_ids function")
    video_ids = []

    for url in url_list:
        if 'youtube.com/watch?v=' in url:
            video_id = url.split('v=')[1].split('&')[0]
        elif 'youtu.be/' in url:
            video_id = url.split('youtu.be/')[1].split('?')[0]
        elif 'youtube.com/shorts/' in url:
            video_id = url.split('youtube.com/shorts/')[1].split('?')[0]
        else:
            raise ValueError("Invalid YouTube URL")

        video_ids.append(video_id)
    logger.info(f"Extracted videos IDs sucessfully")
    return video_ids
    
    
def delete_document_helper(urls:list[str] ,file_names: list[str])->list[str]:
    """
    Processes a list of URLs and file names to determine which documents need to be deleted.

    Args:
        urls (list[str]): A list of URLs to process.
        file_names (list[str]): A list of file names to include for deletion.

    Returns:
        list[str]: A list of document identifiers (URLs or file names) to be deleted.
    """
    logger.debug("Starting delete_document_helper function")
    final_docs_to_delete=[]
    final_docs_to_delete.extend(file_names)
    youtube_links,web_links=classify_urls(urls)
    for youtubeurl in youtube_links:
        if youtubeurl.startswith("@"):
            channel_id=get_channel_id(youtubeurl)
            channel_links=get_channel_videos_links(channel_id)
            videos_id=extract_youtube_video_ids(channel_links)
            final_docs_to_delete.extend(videos_id)
        else:
            final_docs_to_delete.append(youtubeurl)

    for weburl in web_links:
        if is_root_page(weburl):
            final_docs_to_delete.extend(get_all_sitemap_urls(weburl,max_links=300))
        else:
            final_docs_to_delete.append(weburl)
    logger.info(f"Documents to delete: {final_docs_to_delete}")
    return final_docs_to_delete


class DeleteAPIRequest(BaseModel):
    urls:Optional[list[str]]=Field(None,description="List of urls to perfrom the deletion operation from the vector store")
    file_names:Optional[list[str]]=Field(None,description="List of file names to delete from the vector store")
    admin_id:Optional[str]=Field(None,description="bot_id to validate and delete the data from the respective chatbot")



app10=FastAPI()
@app10.post("/1")
async def delete_data(input:DeleteAPIRequest):
    logger.debug("Starting delete_data endpoint")
    try:
        final_docs_to_delete=delete_document_helper(input.urls,input.file_names)
        return delete_documents(final_docs_to_delete,input.admin_id)
    except Exception as e:
        logger.error(f"Error Processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500,detail=f"Error Processing:{str(e)}")
    

from urllib.parse import urlparse,urlunparse
from chatbot.logs import logger
import requests
from typing import List, Optional
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

def is_root_page(url:str)->bool:
    """
    Check if the given URL is a root page URL or a subpage.
    
    Args:
        url (str): The URL to check.
        
    Returns:
        bool: True if the URL is a root root (e.g., kodefast.com), False if it is a subpage.
    """
    logger.info(f"Checking the {url} is root page or an subpage")
    try:
        parsed_url=urlparse(url)
        domain_parts=parsed_url.netloc.split('.')
        is_root=len(domain_parts)==2 and bool(parsed_url.scheme)
        is_subpage=bool(parsed_url.path and parsed_url.path !="/") or parsed_url.query or parsed_url.fragment
        return is_root and not is_subpage
    except Exception as e:
        logger.error(f"Error checking the rootpage:{e}")
        return False
    
def Normalize_url(url:str)->str:
    """
    Convert the given URL into a standardized and accessible format. Adds the scheme if missing,
    removes 'www.' prefix, and ensures the URL is in a consistent format.
    
    Args:
        url (str): The URL to standardize.
        
    Returns:
        str: The standardized URL in a consistent and accessible format.
    
    """
    logger.info(f"Normalizing the url {url} for accessing the data")
    try:
        if not urlparse(url).scheme:
            url = 'https://' + url

        parsed_url=urlparse(url)
        scheme=parsed_url.scheme.lower()
        netloc=parsed_url.netloc.lower()

        if netloc.startswith("www."):
            netloc=netloc[4:]
        path=parsed_url.path.rstrip("/") or "/"

        standardized_url = urlunparse((
            scheme,             
            netloc,             
            path,             
            parsed_url.params,   
            parsed_url.query,  
            parsed_url.fragment  
        ))

        return standardized_url

    
    except Exception as e:
        logger.error(f"Failed to Normalize the Url due the the following error {e}")

def is_url_accessible(url: str, timeout: int = 5) -> bool:
    """
    Check if the given URL is accessible by making an HTTP request.

    Args:
        url (str): The URL to check.
        timeout (int): The timeout for the request in seconds (default is 5 seconds).

    Returns:
        bool: True if the URL is accessible, False otherwise.
    """
    logger.info(f"Started checking the url is accessable or not {url}")
    try:
        response=requests.get(url,timeout=timeout)
        if response.status_code in {200,301,302,403,404}:
            return True
        else:
            logger.info(f"URL not accessible: Status code {response.status_code}")
            return False
    
    except requests.RequestException as e:
        logger.error(f"Error accessing URL: {e}")
        return False
        

class Validateurl(BaseModel):
    url:Optional[str]=Field(None,description="An url to check the validate or not")


#app11=FastAPI()
@app10.post("/2")
async def validate_url(input:Validateurl):
    logger.debug("Starting of the validate_url")
    try:
        normalize_url=Normalize_url(input.url)
        result=is_url_accessible(normalize_url)
        return {"normalize_url":normalize_url,"accessable":result}
    except Exception as e:
        logger.error(f"Error validating the url {e}")
        raise HTTPException(status_code=500,detail=f"Error processing{str(e)}")
        





    


