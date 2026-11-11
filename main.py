
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse, HTMLResponse

# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.models import load_model

# from pydantic import BaseModel, Field
# from contextlib import asynccontextmanager

# import numpy as np
# import pickle
# import re

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("Application starting...")

#     yield

#     print("Application shutting down...")
# app =  FastAPI(lifespan=lifespan)


# """ 
# 1. We are going to make some constrains like
# A. Model Path(BiGRU)
# B. TOKENIZER Path
# C. MAX Sequence Length
# D. Emotion Labels
# E. Emotion emojis
# """
# # A. Model Path(BiGRU)
# model_path = 'Artifact/BiGRU_Model.keras'

# #TOKENIZER Path
# tokenizer_path = 'Artifact/tokenizer.pkl'

# # MAX Sequence Length
# max_sequence_length = 50

# # Emotion Labels
# emotion_labels = ['anger', 'sadness', 'fear', 'joy', 'love ', 'surprise']

# # Emotion emojis
# EMOTION_EMOJIS = {
#     'anger': '😠',
#     'sadness': '😢',
#     'fear': '😨',
#     'joy': '😄',
#     'love': '😍',
#     'surprise': '😲'
# }

# """
# 2.Preprocess the upcoming text
# clean raw text so it matches the format of the training data. This may include lowercasing, removing punctuation, and tokenizing the text.
# A. Lowercasing
# B. Remove apostrophes
# C. Removing Punctuation
# D. Remove extra spaces

# """
# def preprocess_text(text: str) -> str:
#     text = text.lower()
#     text = re.sub(r"'", "", text)  # Remove apostrophes
#     text = re.sub(r"[^a-z0-9\s]", " ", text) # Remove punctuation
#     text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
#     return text



# """ 
# 3.Request and Response Schemas
# A. Text Input -> Input Schema the text sent by user 
# B. Prediction Response -> output schema the emotion to predict
# C. Health Response 
# """
# class TextInput(BaseModel):
#     text : str = Field(
#         ..., 
#         min_length=1,
#         max_length=2000,
#         description = "The sentence to analyze",
#         json_schema_extra = {"example": "I am feeling very happy today!"}
#     )
# class PredictionResponse(BaseModel):
#     text:str
#     predicted_emotion:str
#     confidence:float
#     all_probabilities:dict[str,float]

# class HealthResponse(BaseModel):
#     status:str
#     model_loaded:bool 

# """
# Model Loading and LifeSpan Management
# Load the model and tokenizer when the application starts up

# """
# dl_model = {}
# @asynccontextmanager #jo 2 kaam kar raha
# async def lifespan(app: FastAPI):
#     print("Loading model and tokenizer...")
#     dl_model['BiGRU'] = load_model(model_path)          #BiGRU Model
#     with open(tokenizer_path, 'rb') as file:
#         dl_model['tokenizer'] = pickle.load(file)   #Tokenizer Model
#     print("Model and tokenizer loaded successfully.")

#     yield #pause , model is loaded and server is running and at this point model wait karega req

#     dl_model.clear()  # Clear the model and tokenizer when the application shuts down   

# """
# Mount the static files to the FastAPI ap
# A. Enable CORS (Cross-Origin Resource Sharing) TO allow from different origins.
# """
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"]
# )
# app.mount("/static", StaticFiles(directory="static"), name="static") 
# """
# 6.API Endpoints
# A. Serve UI at homepage ('/')
# B. Health Check Endpoint ('/health')
# C. Predict Emotion Endpoint ('/predict')
# """

# @app.get('/', include_in_schema=False)
# def serve_ui():
#     return FileResponse('static/index.html')

# #b. Health Check Endpoint ('/health')
# @app.get('/health', response_model=HealthResponse)
# def health_check():
#     return HealthResponse(status="server is running", model_loaded=bool(dl_model))

# #C. Predict Emotion Endpoint ('/predict')
# @app.post('/predict', response_model=PredictionResponse)
# def predict_emotion(input_data: TextInput):
#     #1. clean the input text
#     cleaned_text = preprocess_text(input_data.text)
#     #2. convert the words into numeric using tokenizer
#     #3. pad the sequences to ensure uniform length 
#     #4. Run prediction using the BiGRU model
#     #5. Return the top emotion and full probability breakdown

#     BiGRU_model = dl_model['BiGRU']
#     tokenizer_model = dl_model['tokenizer']

#     if BiGRU_model is None or tokenizer_model is None:
#         raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")

#     #1.
#     cleaned_text = preprocess_text(input_data.text)
#     #2 and 3
   
#     tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
#     padded_text = pad_sequences(

#         tokenized_text,
#         maxlen=max_sequence_length,
#         padding='post',
#         truncating='post')
#     probabilities = BiGRU_model.predict(padded_text)[0]
#     top_emotion_index = int(np.argmax(probabilities))
#     all_probabilities = {
#         label: float(prob) for prob, label in zip(probabilities, emotion_labels)

#     }
#     return PredictionResponse(
        
#         text=input_data.text,
#         predicted_emotion=emotion_labels[top_emotion_index],
#         confidence=float(probabilities[top_emotion_index]),
#         all_probabilities=all_probabilities
#     )

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

import numpy as np
import pickle
import re
from pathlib import Path


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model and tokenizer...")
    dl_model['BiGRU'] = load_model(model_path)          #BiGRU Model
    with open(tokenizer_path, 'rb') as file:
        dl_model['tokenizer'] = pickle.load(file)   #Tokenizer Model
    print("Model and tokenizer loaded successfully.")

    yield #pause , model is loaded and server is running and at this point model wait karega req

    dl_model.clear()  # Clear the model and tokenizer when the application shuts down


app = FastAPI(lifespan=lifespan)


""" 
1. We are going to make some constrains like
A. Model Path(BiGRU)
B. TOKENIZER Path
C. MAX Sequence Length
D. Emotion Labels
E. Emotion emojis
"""
# A. Model Path(BiGRU)
model_path = 'Artifact/BiGRU_Model.keras'

#TOKENIZER Path
tokenizer_path = 'Artifact/tokenizer.pkl'

# MAX Sequence Length
max_sequence_length = 50

# Emotion Labels
emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Emotion emojis
EMOTION_EMOJIS = {
    'sadness': '😢',
    'joy': '😄',
    'love': '😍',
    'anger': '😠',
    'fear': '😨',
    'surprise': '😲'
}

"""
2.Preprocess the upcoming text
clean raw text so it matches the format of the training data. This may include lowercasing, removing punctuation, and tokenizing the text.
A. Lowercasing
B. Remove apostrophes
C. Removing Punctuation
D. Remove extra spaces

"""
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)  # Remove apostrophes
    text = re.sub(r"[^a-z0-9\s]", " ", text) # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text



""" 
3.Request and Response Schemas
A. Text Input -> Input Schema the text sent by user 
B. Prediction Response -> output schema the emotion to predict
C. Health Response 
"""
class TextInput(BaseModel):
    text : str = Field(
        ..., 
        min_length=1,
        max_length=2000,
        description = "The sentence to analyze",
        json_schema_extra = {"example": "I am feeling very happy today!"}
    )
class PredictionResponse(BaseModel):
    text:str
    predicted_emotion:str
    confidence:float
    all_probabilities:dict[str,float]

class HealthResponse(BaseModel):
    status:str
    model_loaded:bool 

"""
Model Loading and LifeSpan Management
Load the model and tokenizer when the application starts up

"""
dl_model = {}

"""
Mount the static files to the FastAPI ap
A. Enable CORS (Cross-Origin Resource Sharing) TO allow from different origins.
"""
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static") 

"""
6.API Endpoints
A. Serve UI at homepage ('/')
B. Health Check Endpoint ('/health')
C. Predict Emotion Endpoint ('/predict')
"""

@app.get('/', include_in_schema=False)
def serve_ui():
    return FileResponse(BASE_DIR / 'static' / 'index.html')

#b. Health Check Endpoint ('/health')
@app.get('/health', response_model=HealthResponse)
def health_check():
    return HealthResponse(status="server is running", model_loaded=bool(dl_model))

#C. Predict Emotion Endpoint ('/predict')
@app.post('/predict', response_model=PredictionResponse)
def predict_emotion(input_data: TextInput):
    #1. clean the input text
    cleaned_text = preprocess_text(input_data.text)
    #2. convert the words into numeric using tokenizer
    #3. pad the sequences to ensure uniform length 
    #4. Run prediction using the BiGRU model
    #5. Return the top emotion and full probability breakdown

    BiGRU_model = dl_model['BiGRU']
    tokenizer_model = dl_model['tokenizer']

    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")

    #1.
    cleaned_text = preprocess_text(input_data.text)
    #2 and 3
   
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    print(cleaned_text)
    padded_text = pad_sequences(

        tokenized_text,
        maxlen=max_sequence_length,
        padding='post',
        truncating='post')
    
    probabilities = BiGRU_model.predict(padded_text)[0]
    top_emotion_index = int(np.argmax(probabilities))
    all_probabilities = {
        label: float(prob) for prob, label in zip(probabilities, emotion_labels)

    }
    return PredictionResponse(
        
        text=input_data.text,
        predicted_emotion=emotion_labels[top_emotion_index],
        confidence=float(probabilities[top_emotion_index]),
        all_probabilities=all_probabilities
    )
