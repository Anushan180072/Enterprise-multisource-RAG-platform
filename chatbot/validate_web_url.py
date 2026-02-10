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


app11=FastAPI()
@app11.post("/")
async def validate_url(input:Validateurl):
    logger.debug("Starting of the validate_url")
    try:
        normalize_url=Normalize_url(input.url)
        result=is_url_accessible(normalize_url)
        return {"normalize_url":normalize_url,"accessable":result}
    except Exception as e:
        logger.error(f"Error validating the url {e}")
        raise HTTPException(status_code=500,detail=f"Error processing{str(e)}")
        



