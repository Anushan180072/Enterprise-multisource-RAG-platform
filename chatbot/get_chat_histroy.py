from chatbot.logs import logger
from chatbot.connect_to_mongodb import MongoDBClient
from dotenv import load_dotenv
import os
load_dotenv()
from chatbot.constants import AI_CONNECTION_STRING 
logger.info(f"Mongo url found")
client = MongoDBClient.get_client(AI_CONNECTION_STRING)

db=client["AI-ML"]
collection=db["chatbot-History"]
#print(collection.find_one())
 
def get_chat_history_by_admin_id(admin_id, start_date=None, end_date=None):
    """
    Retrieves the chat history for all session IDs associated with a specific admin_id,
    optionally filtering by a date range.

    Args:
        admin_id (str): The admin ID.
        start_date (datetime, optional): The start date for filtering the chat history.
        end_date (datetime, optional): The end date for filtering the chat history.

    Returns:
        dict: A dictionary where the keys are session IDs and the values are lists of question-answer pairs.
    """
    try:
        logger.debug(f"Starting to retrieve chat history for admin_id: {admin_id}")
        query = {'admin_id': admin_id}
        
        if start_date or end_date:
            date_filter = {}
            if start_date:
                logger.debug(f"Applying start date filter: {start_date}")
                date_filter['$gte'] = start_date
            if end_date:
                logger.debug(f"Applying end date filter: {end_date}")
                date_filter['$lte'] = end_date
            query['chat_history.timestamp'] = date_filter
        
        # Query to find documents with the given admin_id and optional date range
        if start_date and end_date:
            projection = {
                'session_id': 1,
                'chat_history': {
                    '$filter': {
                        'input': '$chat_history',
                        'as': 'entry',
                        'cond': {
                            '$and': [
                                {'$gte': ['$$entry.timestamp', start_date]},
                                {'$lte': ['$$entry.timestamp', end_date]}
                            ]
                        }
                    }
                },
                '_id': 0
            }
        else:
            projection = {
                'session_id': 1,
                'chat_history': 1,
                '_id': 0
            }
        logger.debug(f"Query: {query}, Projection: {projection}")
        results = collection.find(query, projection)
        chat_histories = {}
        
        for result in results:
            session_id = result.get('session_id')
            chat_history = result.get('chat_history', [])
            if chat_history:  # Only add if there's relevant chat history
                chat_histories[session_id] = [
                    {
                        'timestamp': entry['timestamp'],
                        'question': entry['question'],
                        'answer': entry['answer']
                    }
                    for entry in chat_history
                ]
        
        logger.info(f"Successfully retrieved chat history for admin_id: {admin_id}")
        return chat_histories

    except Exception as e:
        logger.error(f"Error retrieving chat history for admin_id is: {admin_id} - {str(e)}")
        return {}


def bot_statistics(data):
    """
    Calculates statistics from the provided chat data.

    Parameters:
    data (dict): A dictionary where the key is a bot ID and the value is another dictionary
                 with user IDs as keys and lists of conversations as values.

    Returns:
    dict: A dictionary where each key is a bot ID and the value is another dictionary with
          'total_sessions' and 'users' statistics. 'users' is a dictionary where the key is
          a user ID and the value is a dictionary with 'conversation_length' indicating the
          number of conversations for that user.
    """
    logger.info("Starting statistics calculation.")
    
    bot_statistics = {}
    
    for bot_id, sessions in data.items():
        #logger.debug(f"Processing bot ID: {bot_id}")
        bot_stats = {
            'total_sessions': len(sessions),
            'users': {}
        }
        
        for user_id, conversations in sessions.items():
            bot_stats['users'][user_id] = {
                'conversation_length': len(conversations)
            }
        
        bot_statistics[bot_id] = bot_stats
        logger.debug(f"Calculated stats for bot ID: {bot_id} - {bot_stats}")
    
    logger.info("Statistics calculation completed.")
    return bot_statistics

def get_all_bots_chat(admin_ids:list[str],start_date=None,end_date=None)->dict[dict]:
    """
    Retrieves the chat history for a list of admin IDs within an optional date range.

    Args:
        admin_ids (list[str]): A list of admin IDs for which to retrieve chat histories.
        start_date (datetime, optional): The start date for filtering the chat histories.
        end_date (datetime, optional): The end date for filtering the chat histories.

    Returns:
        list[dict]: A list of dictionaries, each containing the chat history for a specific admin ID.
    """
    logger.debug(f"Starting to retrieve chat histories for admin_ids: {admin_ids}")
    final_chat = {}
    try:
        for admin_id in admin_ids:
            logger.debug(f"Retrieving chat history for admin_id: {admin_id}")
            final_chat[admin_id]=get_chat_history_by_admin_id(admin_id,start_date,end_date)                
            logger.info(f"Successfully retrieved chat history for admin_id: {admin_id}")
            
    except Exception as e:
        logger.error(f"Error while retrieving chat histories for admin_ids: {admin_ids} - {str(e)}", exc_info=True)

    logger.debug(f"Completed retrieval for all admin_ids")
    return final_chat

def delete_sessions(admin_id, session_id):
    """
    Delete the chat history for a session, based on admin_id and session_id.

    Args:
        admin_id (str): Admin ID for which to delete the chat history.
        session_id (str): Session ID for which to delete the chat history.

    Returns:
        boolean, int: A success message(True) with the number of documents deleted, or an error message if the operation fails.
    """
    try:
        query = {
            "admin_id": admin_id,
            "session_id": session_id
        }
        result = collection.delete_many(query)
        deleted_count = result.deleted_count
        
        logger.info(f"{deleted_count} chat(s) deleted.")
        return True, deleted_count
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return False, 0



    


