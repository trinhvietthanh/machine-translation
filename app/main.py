import pickle
from typing import Optional
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class Models(BaseModel):
    lang: str
    modelName: str 

MODEL = 'best_vi_translation_v1_model.augmented.vien.ckpt-1547000.data-00000-of-00001'
vocab_file = 'vocab.subwords'
model_list ={
    'smt-en-vi': 'smt_envi.ptk',
    'smt-vi-en': 'smt_vien.ptk',
    'transformer-vi-en': '',
    'transformer-en-vi': '',
}
app = FastAPI()


@app.get("/")
def index():
    return {"get": "World"}


@app.post('get_config')
def config(data:Models):
    data = data.dict()
    
    model_path = "/models/{}".format(data.modelName)
    
