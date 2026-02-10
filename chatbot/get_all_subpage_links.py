import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import random
from chatbot.logs import logger

# List of different User-Agents for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/89.0 Safari/537.36',
]

# Optional: Proxy list for bypassing IP restrictions
PROXIES = [
    # 'http://proxy1.com:port',
    # 'http://proxy2.com:port',
]

def fetch_page(url: str, session: requests.Session, max_links: int, collected_links: set) -> set:
    """
    Fetches the given URL and parses it for links.

    Parameters:
    url (str): The URL to fetch.
    session (requests.Session): The session object for making requests.
    max_links (int): The maximum number of links to collect.
    collected_links (set): A set of already collected links.

    Returns:
    set: A set of all unique subpage URLs found within the parsed HTML.
    """
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    proxy = random.choice(PROXIES) if PROXIES else None
    try:
        response = session.get(url, headers=headers, proxies={"http": proxy, "https": proxy}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        links = [urljoin(url, link.get('href')) for link in soup.find_all('a', href=True)]
        subpage_links = set(link for link in links if urlparse(link).netloc == urlparse(url).netloc)
        
        if not subpage_links:  # If no sublinks found, return the original URL
            subpage_links.add(url)

        # Add links to collected_links until max_links is reached
        for link in subpage_links:
            if len(collected_links) >= max_links:
                logger.info(f"Reached max_links limit: {max_links}. Stopping further collection.")
                return collected_links
            collected_links.add(link)
        
        logger.info(f"Fetched {len(subpage_links)} links from {url}")
        return collected_links
    except requests.RequestException as e:
        logger.error(f"Failed to retrieve page: {url} with error: {e}")
        return collected_links
    


def fetch_robots_txt(root_url: str, session: requests.Session) -> list:
    """
    Fetches the robots.txt file from the root URL and extracts sitemap URLs.

    Parameters:
    root_url (str): The root URL to fetch robots.txt from.
    session (requests.Session): The session object for making requests.

    Returns:
    list: A list of sitemap URLs found in the robots.txt.
    """
    sitemap_urls = []
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    proxy = random.choice(PROXIES) if PROXIES else None
    try:
        robots_url = urljoin(root_url, '/robots.txt')
        response = session.get(robots_url, headers=headers, proxies={"http": proxy, "https": proxy}, timeout=10)
        response.raise_for_status()

        for line in response.text.splitlines():
            if line.lower().startswith('sitemap:'):
                sitemap_url = line.split(':', 1)[1].strip()
                sitemap_urls.append(sitemap_url)
        
        logger.info(f"Found {len(sitemap_urls)} sitemaps in robots.txt from {root_url}")
    except requests.RequestException as e:
        logger.error(f"Failed to retrieve robots.txt: {e}")
    
    return sitemap_urls



def fetch_sitemap_urls(sitemap_urls: list, session: requests.Session, max_links: int, collected_links: set, visited_sitemaps: set = None) -> set:
    """
    Fetches all URLs listed in the sitemap(s), with a limit of max_links.

    Parameters:
    sitemap_urls (list): List of sitemap URLs to fetch.
    session (requests.Session): The session object for making requests.
    max_links (int): The maximum number of links to collect.
    collected_links (set): A set of already collected links.
    visited_sitemaps (set): A set of already visited sitemaps to avoid duplication.

    Returns:
    set: A set of all unique subpage URLs found in the sitemap(s).
    """
    if visited_sitemaps is None:
        visited_sitemaps = set()
        
    headers = {'User-Agent': random.choice(USER_AGENTS)}
    proxy = random.choice(PROXIES) if PROXIES else None
    
    for sitemap_url in sitemap_urls:
        if sitemap_url in visited_sitemaps or len(collected_links) >= max_links:
            continue

        try:
            visited_sitemaps.add(sitemap_url)
            response = session.get(sitemap_url, headers=headers, proxies={"http": proxy, "https": proxy}, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'xml')
            urls = [loc.text for loc in soup.find_all('loc')]
            
            # Check if any of these URLs are additional sitemaps
            if any(url.endswith('.xml') for url in urls):
                nested_sitemap_urls = [url for url in urls if url.endswith('.xml')]
                fetch_sitemap_urls(nested_sitemap_urls, session, max_links, collected_links, visited_sitemaps)
            else:
                for url in urls:
                    if len(collected_links) >= max_links:
                        logger.info(f"Reached max_links limit: {max_links}. Stopping further collection.")
                        return collected_links
                    collected_links.add(url)
                
            logger.info(f"Fetched {len(urls)} URLs from sitemap {sitemap_url}")
            
        except requests.RequestException as e:
            logger.error(f"Failed to retrieve or parse sitemap: {sitemap_url} with error: {e}")
    
    return collected_links


def crawl_website(root_url: str, max_links: int) -> set:
    """
    Crawls the entire website starting from the root URL to find all subpages.

    Parameters:
    root_url (str): The root URL to start crawling from.
    max_links (int): The maximum number of links to collect.

    Returns:
    set: A set of all unique subpage URLs found within the website.
    """
    root_url = root_url if root_url.startswith(('http://', 'https://')) else 'http://' + root_url
    root_domain = urlparse(root_url).netloc
    visited = set()
    to_visit = {root_url}
    session = requests.Session()  # Start a session to persist headers and cookies
    collected_links = set()
    collected_links.add(root_url)

    while to_visit and len(collected_links) < max_links:
        current_url = to_visit.pop()
        if current_url not in visited:
            visited.add(current_url)
            fetch_page(current_url, session, max_links, collected_links)
            for link in collected_links:
                if len(collected_links) >= max_links:
                    logger.info(f"Reached max_links limit: {max_links}. Stopping further collection.")
                    return collected_links
                if link not in visited and urlparse(link).netloc == root_domain:
                    to_visit.add(link)

    logger.info(f"Crawled {len(collected_links)} links from {root_url}")
    return collected_links

def get_all_sitemap_urls(root_url: str, max_links: int ) -> list:
    """
    Retrieves all sitemap URLs from the robots.txt and processes them.
    If no sitemaps are found, parses the entire website for subpages.

    Parameters:
    root_url (str): The root URL to start fetching sitemaps from.
    max_links (int): The maximum number of links to collect.

    Returns:
    list: A list of all unique subpage URLs found.
    """
    root_url = root_url if root_url.startswith(('http://', 'https://')) else 'http://' + root_url
    session = requests.Session()  # Start a session to persist headers and cookies
    collected_links = set()

    # Fetch sitemap URLs from robots.txt
    sitemap_urls = fetch_robots_txt(root_url, session)
    
    # Fetch URLs from each sitemap
    if sitemap_urls:
        all_sitemap_urls = fetch_sitemap_urls(sitemap_urls, session, max_links, collected_links)
    else:
        logger.info("No sitemaps found in robots.txt. Crawling the entire website.")
        all_sitemap_urls = crawl_website(root_url, max_links)
    
    return list(all_sitemap_urls)
