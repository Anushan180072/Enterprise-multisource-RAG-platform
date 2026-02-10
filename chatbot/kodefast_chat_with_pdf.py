from chatbot.add_user_data_to_vectordb import vector_store,atlas_collection,get_text_chunks,add_admin_id_to_docs
from chatbot.data_extract import process_s3_urls
from typing import List,Optional,Tuple
from langchain.schema import Document
from chatbot.logs import logger
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel,Field
from fastapi import FastAPI
from typing import Any,List
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from chatbot.chat_bot import google_llm
from chatbot.add_user_data_to_vectordb import store_user_all_data_with_id,vector_store,get_text_chunks,add_admin_id_to_docs,preprocess_documents

from fastapi import HTTPException

class generate_question(BaseModel):
    data:str=Field(description="The data from where we need to generate the questions")

parser=JsonOutputParser()
response_parser=parser.get_format_instructions()

prompt = PromptTemplate(
    template="""You are an expert at generating questions from the given input. Given the following text, generate exactly five questions that could be asked about it. The questions should be concise, relevant, and directly related to the input text,ensure that dont give any  " "

    Example:
    Text: "The Industrial Revolution was a period of major industrialization that took place during the late 1700s and early 1800s. It began in Great Britain and quickly spread throughout the world."
    
    Expected Output:
    [
         What were the key factors that led to the Industrial Revolution?
         How did the Industrial Revolution spread globally?
         What were the major impacts of the Industrial Revolution on society?
    ]

    Now, generate questions for the following text by following the {format_instructions}.

    Text: {input_text}

    Provide the questions in a JSON format as shown above.
    """,
    input_variables=["input_text"],
    partial_variables={"format_instructions": response_parser},
)


question_chain = prompt | google_llm | StrOutputParser()


def generate_questions(input_texts: list[str]) -> str:
    """
    Generates three questions based on the combined input strings.

    Parameters:
    - input_texts (list[str]): A list of input strings to generate questions from.

    Returns:
    - str: The generated questions.
    """
    if not input_texts or all(not text.strip() for text in input_texts):
        logger.warning("No input text provided for questions generation.")
        return "[]"
    combined_text = " ".join(input_texts)
    
    # Pass the combined string to the question chain and generate questions
    return question_chain.invoke(combined_text)


def extract_page_content(documents: List[Document]) -> List[str]:
    """
    Extracts the `page_content` from a list of LangChain `Document` objects.

    Args:
        documents (List[Document]): A list of LangChain `Document` objects.

    Returns:
        List[str]: A list of `page_content` strings extracted from the provided documents.
    """
    try:
        logger.info("Starting to extract page content from documents.")
        
        # Extracting page content
        page_contents = [doc.page_content for doc in documents]
        
        logger.info(f"Successfully extracted page content from {len(documents)} documents.")
        return page_contents
    
    except Exception as e:
        logger.error(f"An error occurred while extracting page content: {str(e)}")
        raise


def preprocessing_questions(input_list: list)-> list:
    new_list=[]
    res=input_list.split("\n")
    for i in res[2:5]:
        new_list.append(i.strip().strip('",'))

    return new_list





def store_kodefast_pdf_chat_data(admin_id:str,s3_file_ulr: List[str]):
    """
    Process a list of S3 URLs and add metadata including admin_id.

    Args:
        urls (List[str]): List of S3 URLs.
        admin_id (str): Admin ID to add as metadata.

    Returns:
        Dict: A dictionary with the processed results.
    """
    kodefast_pdf_chat_data=[]
    raw_data_from_s3url=process_s3_urls(s3_file_ulr)
    kodefast_pdf_chat_data.extend(raw_data_from_s3url)
    trimmed_data=extract_page_content(raw_data_from_s3url)
    #print(trimmed_data)
    if len(trimmed_data)>5:
        llm_generate_questions=generate_questions(trimmed_data[0:3])
    else:
        llm_generate_questions=generate_questions(trimmed_data)
    preprocessed_question=preprocessing_questions(llm_generate_questions)
    chunked_data=get_text_chunks(kodefast_pdf_chat_data)
    id_added_data=add_admin_id_to_docs(chunked_data,admin_id)
    vector_store.add_documents(id_added_data)
    return preprocessed_question

from chatbot.kodefast_chat_with_pdf import store_kodefast_pdf_chat_data
class kodefastAPIrequest(BaseModel):
    admin_id:Optional[str] = Field(None, description="Unique identifier for the user")
    fileUrls: Optional[List[str]] = Field(None, description="List of file download URLs")


app13=FastAPI()

@app13.post("/")
async def ingest_kodefast_pdf_chat(input:kodefastAPIrequest):
    try:
        result=store_kodefast_pdf_chat_data(input.admin_id,input.fileUrls)
        if result:
            return result
        else:
            return {"message": "No questions were generated because the input text was empty or insufficient."}
    except Exception as e:
        logger.error(f"Error during data ingestion for bot ID: {input.admin_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")









