import pandas as pd
from data.db import DATA
from fastapi import FastAPI
from fastapi.responses import FileResponse
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


data = pd.DataFrame()
data["text"] = DATA["title"].astype(str) + DATA["description"].astype(str)
data["text"] = data["text"].apply(lambda x: x.lower())
data["text"] = data["text"].str.replace("[^\w\s]", "")
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(data["text"])
knn = NearestNeighbors(n_neighbors=10, metric="cosine")
knn.fit(tfidf_matrix)
app = FastAPI()


@app.get("/")
async def get_search_form():
    return FileResponse("index.html")


@app.get("/search")
def search(q: str):
    query_tfidf = vectorizer.transform([q])
    try:
        _, top_indices = knn.kneighbors(query_tfidf)
        top_texts = DATA.iloc[top_indices[0]]["title"].tolist()
        top_description = DATA.iloc[top_indices[0]]["description"].tolist()
    except ValueError:
        top_texts, top_description = [], []
    return {
        "results": [
            {"title": x, "description": y} for x, y in zip(top_texts, top_description)
        ]
    }
