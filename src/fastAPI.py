import uvicorn
from fastapi import FastAPI
from .models.DecisionTree import DecisionTree
from pydantic import BaseModel
from typing import Optional
from .models.utils import dict_to_pandas


app = FastAPI()
dt_pipeline = DecisionTree()

class Music(BaseModel):
    acousticness: float
    danceability: float
    energy: float
    valence: float
    key: int
    loudness:float
    name: str
    explicit: bool
    artists: Optional[str]
    mode: Optional[int]
    popularity: Optional[int]

@app.get("/")
def hello_word():
    return {"message":"Hello Word"}

@app.post("/predict/")
async def predict_music(music:Music):
    columns_to_use = ['acousticness', 'danceability', 'energy', 'valence', 'explicit', 'key', 'loudness']
    df_to_predict = dict_to_pandas(music.dict(), columns_to_use)
    popularity = dt_pipeline.predict(df_to_predict)[0]
    response_body = {}
    response_body["name"] = music.dict()["name"]
    response_body["body_request"] = music.dict()
    response_body["predicted_popularity"] = popularity
    return response_body

# @app.post("/predict/list")
# async def predict_list(data: MusicList):
#     pass

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)
