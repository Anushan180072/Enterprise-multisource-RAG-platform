from fastapi import FastAPI
import warnings
warnings.filterwarnings("ignore")

from chatbot.chat_bot import app6
#from chatbot.nimble import app7
from chatbot.add_user_data_to_vectordb import app9
from chatbot.delete_user_files import app10
from chatbot.validate_web_url import app11
from chatbot.kodefast_chat_with_pdf import app13


from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.wsgi import WSGIMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/chatbot", app6)
#app.mount("/nimble_property", app7)
app.mount("/data_ingestion",app9)
app.mount("/delete_data",app10)
app.mount("/validate_url",app11)
app.mount("/kodefast_chat",app13)


