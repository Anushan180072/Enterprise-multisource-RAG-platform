import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import List
from langchain_community.document_loaders import YoutubeLoader
from langchain.schema import Document
from chatbot.logs import logger

def get_channel_id(channel_name: str) -> str:
    """
    Retrieves the YouTube channel ID for a given channel name.
    
    Args:

        channel_name (str): The name or username of the channel to search for.
        
    Returns:
        str: The channel ID if found, otherwise an error message.
    """
    logger.info(f"Started get_channel_id for channel_name: {channel_name}")
    search_url = "https://www.googleapis.com/youtube/v3/search"
    api_key = "AIzaSyA0LS9feIKZ1vVFLUP4KJgS1oFqjFPWxGo" 
    params = {
        "part": "snippet",
        "q": channel_name,
        "type": "channel",
        "key": api_key,
        "maxResults": 1
    }
    
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()

        data = response.json()

        if "items" in data and len(data["items"]) > 0:
            channel_id = data["items"][0]["snippet"]["channelId"]
            logger.info(f"Channel ID found: {channel_id}")
            return channel_id
        else:
            logger.warning(f"Channel not found for: {channel_name}")
            return "Channel not found."
        
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error occurred: {conn_err}")
        return f"Connection error occurred: {conn_err}"
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Timeout error occurred: {timeout_err}")
        return f"Timeout error occurred: {timeout_err}"
    except requests.exceptions.RequestException as req_err:
        logger.error(f"An error occurred: {req_err}")
        return f"An error occurred: {req_err}"
    finally:
        logger.info(f"Finished get_channel_id for channel_name: {channel_name}")
    



def get_channel_videos_links(channel_id: str) -> List[str]:
    """
    Fetches all video links from a YouTube channel's upload playlist.

    Args:
        channel_id (str): The ID of the YouTube channel.

    Returns:
        List[str]: A list of video URLs from the channel.
    """

    API_KEY = 'AIzaSyA0LS9feIKZ1vVFLUP4KJgS1oFqjFPWxGo'  
    YOUTUBE_API_SERVICE_NAME = 'youtube'
    YOUTUBE_API_VERSION = 'v3'
    
    try:
    
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

        channel_response = youtube.channels().list(
            part='contentDetails',
            id=channel_id
        ).execute()

        if 'items' not in channel_response or not channel_response['items']:
            logger.warning(f"No content details found for channel_id: {channel_id}")
            return []

        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']


        video_links = []
        next_page_token = None

        while True:
            playlist_response = youtube.playlistItems().list(
                part='snippet',
                playlistId=uploads_playlist_id,
                maxResults=50,  
                pageToken=next_page_token
            ).execute()

            for item in playlist_response['items']:
                video_id = item['snippet']['resourceId']['videoId']
                video_links.append(f'https://www.youtube.com/watch?v={video_id}')

            logger.info(f"Retrieved {len(playlist_response['items'])} videos from playlist.")


            next_page_token = playlist_response.get('nextPageToken')

            if not next_page_token:
                break
        logger.info(f"Total videos retrieved: {len(video_links)} for channelID: {channel_id}" )
        return video_links

    except HttpError as e:
        logger.error(f"An HTTP error occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

    





def process_youtube_links(result_links: list[str])->list[Document]:
    """
    Process a list of YouTube URLs to load video data using the YoutubeLoader.
    
    Args:
        result_links (list): A list of YouTube video URLs.
    
    Returns:
        dict: A dictionary with 'data' containing loaded documents and 'errors' containing error messages.
    """
    logger.info(f"Started process_youtube_links with {len(result_links)} links")
    data = []
    errors = []

    for url in result_links:
        logger.info(f"Processing URL: {url}")
        try:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
            documents = loader.load()
            data.extend(documents)
            logger.info(f"Successfully processed URL: {url}")
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            errors.append({'url': url, 'error': str(e)})

    logger.info(f"Finished processing links. Total documents: {len(data)}, Errors: {len(errors)}")
    return data

