from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.model import ModelService

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model when app starts
@app.on_event("startup")
def startup():
    ModelService.load()

# Include your routes
app.include_router(router)

@app.get("/")
def home():
    return {"message": "API is running!"}