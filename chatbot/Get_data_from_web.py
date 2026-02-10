from typing import List,Tuple
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
from chatbot.logs import logger
from chatbot.validate_web_url import is_root_page
from chatbot.get_all_subpage_links import get_all_sitemap_urls
from urllib.parse import urljoin, urlparse
import concurrent.futures
import time

MAX_URLS_TO_CRAWL = 150
MAX_WORKERS = 5
BATCH_SIZE = 20

def fetch_webpage(url: str, seen_urls: set[str]) -> List[Document]:
    """
    Concurrently fetches and loads a webpage and its sub-links into Langchain Documents.

    Args:
        url (str): The base URL to crawl.

    Returns:
        List[Document]: List of Langchain Document objects.
    """
    logger.info(f"Fetching webpage and sub-links from: {url}")
    documents = []

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Find internal links
        base_domain = urlparse(url).netloc
        internal_links = {
            urljoin(url, a['href'])
            for a in soup.find_all("a", href=True)
            if urlparse(urljoin(url, a['href'])).netloc == base_domain
        }

        all_urls = list(internal_links)
        all_urls.insert(0, url)
        all_urls = [u for u in all_urls if u not in seen_urls]
        seen_urls.update(all_urls)

        # Limit total URLs per page
        all_urls = all_urls[:MAX_URLS_TO_CRAWL]

        logger.info(f"Found {len(all_urls)} internal links on {url}")

        # Concurrent fetching using ThreadPoolExecutor
        def load_url(link):
            try:
                loader = WebBaseLoader(link, continue_on_failure=True)
                return loader.load()
            except Exception as e:
                logger.warning(f"Failed to load content from {link}: {e}")
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            documents_batches = []
            for i in range(0, len(all_urls), BATCH_SIZE):
                batch = all_urls[i:i + BATCH_SIZE]
                documents_batches.extend(executor.map(load_url, batch))

        for docs in documents_batches:
            documents.extend(docs)

    except Exception as e:
        logger.error(f"Error crawling {url}: {e}")

    return documents

    

def filter_scrapable_urls(urls: List[str]) -> Tuple[List[str], List[str]]:
    """
    Filters a list of URLs into scrapable and non-scrapable categories based on file extensions.

    Args:
        urls (list): A list of URLs to be checked.

    Returns:
        tuple: Two lists of URLs - one for scrapable and one for non-scrapable URLs.
    """
    non_scrapable_extensions = [
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.txt', '.csv', '.rtf',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.tiff', '.ico', '.webp', '.heic',
        '.mp4', '.mkv', '.mov', '.avi', '.flv', '.wmv', '.webm', '.mpeg', '.mpg',
        '.mp3', '.wav', '.aac', '.flac', '.ogg', '.wma', '.m4a', '.opus',
        '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2',
        '.exe', '.dll', '.bin', '.iso', '.dmg', '.apk'
    ]
    logger.info("Filtering URLs into scrapable and non-scrapable categories")
    scrapable_urls = []
    non_scrapable_urls = []

    for url in urls:
        # Extract the path from the URL and check the extension
        path = url.split('?')[0]  # Ignore query parameters when checking the extension
        if any(path.lower().endswith(ext) for ext in non_scrapable_extensions):
            non_scrapable_urls.append(url)
        else:
            scrapable_urls.append(url)
    logger.info(f"Filtered URLs - Scrapable: {len(scrapable_urls)}, Non-Scrapable: {len(non_scrapable_urls)}")
    return scrapable_urls, non_scrapable_urls

    

def fetch_sitemap(url: str, seen_urls: set[str]) -> List[Document]:
    """
    Fetches documents from all subpages listed in the sitemap.

    Args:
        url (str): The URL of the sitemap or base URL.

    Returns:
        List[Document]: A list of documents fetched from the sitemap.
    """
    logger.info(f"Fetching sitemap and documents from URL: {url}")
    data = []
    try:
        sub_links=get_all_sitemap_urls(url,max_links=MAX_URLS_TO_CRAWL)
        scrapable, _ = filter_scrapable_urls(sub_links)
        scrapable = [u for u in scrapable if u not in seen_urls]

        for sub_url in scrapable:
            docs = fetch_webpage(sub_url, seen_urls)
            data.extend(docs)

        logger.info(f"Fetched {len(data)} documents from sitemap URL: {url}")
        return data

    except Exception as e:
        logger.error(f"Error loading sitemap {url}: {e}")
        return []

def extract_text_from_urls(urls: List[str]) -> List[Document]:
    """
    Extracts text from a list of URLs, handling both sitemaps and individual pages.

    Args:
        urls (List[str]): A list of URLs to be processed.

    Returns:
        List[Document]: A list of documents extracted from the URLs.
    """
    data = []
    logger.info("Extracting text from webpage URLs")
    seen_urls: set[str] = set()

    for url in urls:
        logger.info(f"url{url}")
        try:   
            if is_root_page(url):
                logger.info(f"URL {url} considered as sitemap.")
                sitemap_data = fetch_sitemap(url, seen_urls)
                if sitemap_data:
                    data.extend(sitemap_data)
            else:
                logger.info(f"URL {url} considered as individual webpage.")
                webpage_data = fetch_webpage(url, seen_urls)
                #logger.info("webpage_datawebpage_data"+ str(webpage_data))
                if webpage_data:
                    data.extend(webpage_data)
                    #logger.info("dataaaa"+ str(data))
        except Exception as e:
            logger.warning(f"Skipping {url} due to error: {e}")
    logger.info(f"Extracted {len(data)} documents from webpage URLs")
    return data