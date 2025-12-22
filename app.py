from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import json
import uvicorn

# Model and word index
model_params = {}
word_index = {}

# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_params, word_index
    print("Loading model parameters and word index...")

    with np.load('model_params.npz', allow_pickle=True) as data:
        model_params['W1'] = data['W1'].T
        model_params['b1'] = data['b1']
        model_params['W2'] = data['W2'].T
        model_params['b2'] = data['b2']
        model_params['W3'] = data['W3'].T
        model_params['b3'] = data['b3']

    with open('word_index.json', 'r') as f:
        word_index = json.load(f)

    print("Startup complete.")
    yield
    print("Shutdown complete.")

# ---------- App ----------
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Forward Pass ----------
def forward_pass(text: str) -> float:
    W1 = model_params['W1']
    b1 = model_params['b1']
    W2 = model_params['W2']
    b2 = model_params['b2']
    W3 = model_params['W3']
    b3 = model_params['b3']

    MAX_WORDS = 10000
    vectorized_text = np.zeros(MAX_WORDS)
    for word in text.split():
        if word in word_index:
            index = word_index[word] + 3
            if index < MAX_WORDS:
                vectorized_text[index] = 1

    h1 = np.dot(vectorized_text, W1) + b1
    a1 = np.maximum(0, h1)  # ReLU
    h2 = np.dot(a1, W2) + b2
    a2 = np.maximum(0, h2)  # ReLU
    h3 = np.dot(a2, W3) + b3
    output = 1 / (1 + np.exp(-h3))  # Sigmoid

    # Robust fix: always reduce to a single float
    return float(np.mean(output))

# ---------- API ----------
@app.post("/predict_sentiment/")
async def predict_sentiment(request: Request):
    try:
        data = await request.json()
        review_text = data.get("review")

        if review_text:
            prediction = forward_pass(review_text)
            sentiment = "positive" if prediction > 0.5 else "negative"
            return {"sentiment": sentiment, "probability": prediction}
        else:
            return {"error": "No review text provided."}
    except Exception as e:
        return {"error": str(e)}

# ---------- Frontend ----------
@app.get("/")
def serve_index():
    return FileResponse("index.html")

app.mount("/", StaticFiles(directory=".", html=True), name="static")

# ---------- Run ----------
if __name__ == "__main__":
    # Run as: python app.py
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
