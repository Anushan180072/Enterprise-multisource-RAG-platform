
from typing import Union
import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from chatbot.logs import logger



def initialize_llm_async(model_name: str, api_key: str):
    """
    Asynchronously initializes and returns an LLM model based on the provided model name and API key.

    Supported Models:
    - GPT-4o, GPT-4o-mini, GPT-3.5-turbo from OpenAI
    - Gemini-2.0-flash from Gemini family
    - Llama3-70b-8192 from Groq family

    Args:
        model_name (str): The name of the LLM model. Supported values:
            - 'gpt-4o'
            - 'gpt-4o-mini'
            - 'gpt-3.5-turbo'
            - 'gemini-2.0-flash'
            - 'llama3-70b-8192'
        api_key (str): The API key for the LLM model.

    Returns:
        Union[ChatOpenAI, ChatGroq, GoogleGenerativeAI]: The initialized LLM model instance.

    Raises:
        ValueError: If the provided model name is not supported.
    """
    
    model_name = model_name.lower()

    
    if model_name in ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']:
        logger.info(f"Initializing OpenAI model asynchronously: {model_name}")
        llm_openai=ChatOpenAI(model=model_name, api_key=api_key)
        return llm_openai
    
    
    elif model_name == 'gemini-2.0-flash':
        logger.info(f"Initializing Gemini model asynchronously: {model_name}")
        llm_google= GoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0, verbose=True, timeout=600)
        return llm_google

    
    elif model_name == 'llama3-70b-8192':
        logger.info(f"Initializing Groq model asynchronously: {model_name}")
        llm_groq=ChatGroq(groq_api_key=api_key, model_name=model_name)
        return llm_groq
    else:
        logger.error(f"Unsupported model name: {model_name}")
        return None