from urllib.parse import urlparse
from typing import List, Tuple
from chatbot.logs import logging


def classify_urls(urls: List[str]) -> Tuple[List[str], List[str]]:
    """
    Classifies a list of URLs into YouTube URLs and generic website URLs.

    Parameters:
    - urls (List[str]): A list of URLs to classify.

    Returns:
    - Tuple[List[str], List[str]]: Two lists, one with YouTube URLs and the other with website URLs.
    """
    youtube_domains = {'www.youtube.com', 'youtube.com', 'm.youtube.com', 'youtu.be'}
    youtube_urls = []
    website_urls = []

    for url in urls:
        parsed_url = urlparse(url)
        if parsed_url.netloc in youtube_domains or parsed_url.path.startswith('@'):
            print(parsed_url.netloc)
            youtube_urls.append(url)
        else:
            website_urls.append(url)

    return youtube_urls, website_urls

