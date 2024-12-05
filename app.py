import pickle
from typing import List, Dict
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .GCN import GCN, smile_to_graph
from torch_geometric.data import Data
import os
import logging
from dotenv import load_dotenv

# 配置日誌
logging.basicConfig(
    level=logging.DEBUG if os.getenv("ENV") == "development" else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{os.getenv('ENV', 'development')}.log")
    ]
)
logger = logging.getLogger(__name__)

# 載入 .env 文件
load_dotenv()

app = FastAPI(
    title="Ecotoxicology Prediction API",
    description="API for predicting ecotoxicology using GCN models",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENV") == "development" else None  # 生產環境禁用 Swagger
)

# 環境變數設置
ENV = os.getenv("ENV", "development")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8890"))
DEBUG = ENV == "development"

logger.info(f"Starting application in {ENV} mode")

try:
    # Load the models
    models = {
        "A2A": pickle.load(open("A2A.pickle", "rb")),
        "C2C": pickle.load(open("C2C.pickle", "rb")),
        "F2F": pickle.load(open("F2F.pickle", "rb")),
    }
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

class PredictionRequest(BaseModel):
    fasta: str
    model_type: str

def parse_fasta(fasta_str: str) -> List[str]:
    entries = fasta_str.split(">")
    sequences = [entry.split("\n", 1)[1].replace("\n", "") for entry in entries if entry]
    return sequences

@app.post("/predict/")
async def predict(request: PredictionRequest):
    logger.debug(f"Received prediction request for model type: {request.model_type}")
    try:
        model = models.get(request.model_type)
        if not model:
            logger.warning(f"Invalid model type requested: {request.model_type}")
            raise HTTPException(status_code=400, detail="Invalid model type")

        sequences = parse_fasta(request.fasta)
        predictions = []

        for seq in sequences:
            data = smile_to_graph(seq)
            output = model(data)
            pred_class = torch.argmax(output, dim=1).item()
            predictions.append({"sequence": seq, "prediction": pred_class})

        logger.info(f"Successfully processed {len(sequences)} sequences")
        return {"predictions": predictions}

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {"status": "healthy", "environment": ENV}

if __name__ == "__main__":
    import uvicorn

    # 開發模式配置
    if DEBUG:
        logger.info("Starting server in development mode")
        uvicorn.run(
            "app:app",
            host=HOST,
            port=PORT,
            reload=True,
            workers=1,
            log_level="debug"
        )
    else:
        # 生產模式配置
        logger.info("Starting server in production mode")
        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            workers=4,
            access_log=True,
            log_level="info"
        )
