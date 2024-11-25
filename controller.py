import io
import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class SoilConditions(BaseModel):
    N: int
    P: int
    K :int
    temperature : float
    humidity :float
    ph :float
    rainfall:float

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods= ["*"], 
    allow_headers=["*"],
)


@app.get("/predict-crop/")
async def predict_crop(data: SoilConditions):
    with open('cp_model.pkl', 'rb') as f:
        try:
            model = pickle.load(f)
            arr = np.array([[data.N,data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]])
            prediction = model.predict(arr)
            print(prediction[0])
            print(returnLabeledCropData(prediction[0]))
            return {"prediction":returnLabeledCropData(prediction[0])}
        except:
            return {"error":"Something error occured"}
        

def returnLabeledCropData(value: float):
    data = int(value)
    match data:
        case 1:
            return "rice"
        case 2:
            return "maize"
        case 3:
            return "jute"
        case 4:
            return "cotton"
        case 5:
            return "papaya"
        case 6:
            return "orange"
        case 7:
            return "apple"
        case 8:
            return "muskmelon"
        case 9:
            return "watermelon"
        case 10:
            return "grapes"
        case 11:
            return "mango"
        case 12:
            return "banana"
        case 13:
            return "pomegranate"
        case 14:
            return "lentil"
        case 15:
            return "blackgram"
        case 16:
            return "mungbean"
        case 17:
            return "mothbeans"
        case 18:
            return "pigeonpeas"
        case 19:
            return "kidneybeans"
        case 20:
            return "chickpea"
        case 21:
            return "coffee"