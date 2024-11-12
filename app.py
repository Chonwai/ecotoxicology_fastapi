from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from service.GCN import load_model, predict_toxicity
import logging
import uvicorn

app = FastAPI()


# 定義接收的 JSON 結構
class PredictionRequest(BaseModel):
    fasta: str
    model_type: str  # 用於指定 F2F, C2C 或 A2A 模型


def validate_fasta(fasta_str):
    fasta_str = fasta_str.replace("\r\n", "\n").replace("\r", "\n")
    lines = fasta_str.strip().split("\n")
    if len(lines) % 2 != 0:
        raise ValueError(
            "FASTA format error: Each ID line must be followed by a sequence line."
        )
    for i in range(0, len(lines), 2):
        if not lines[i].startswith(">"):
            raise ValueError(f"FASTA format error: Line {i+1} does not start with '>'.")
        if not lines[i + 1]:
            raise ValueError(f"FASTA format error: Line {i+2} is empty.")
    return True


def parse_fasta(fasta_str):
    fasta_str = fasta_str.replace("\r\n", "\n").replace("\r", "\n")
    lines = fasta_str.strip().split("\n")
    ids = []
    smiles = []
    for i in range(0, len(lines), 2):
        ids.append(lines[i][1:])  # 去掉 '>'
        smiles.append(lines[i + 1])
    return ids, smiles


@app.post("/api/predict")
def predict_toxicity_endpoint(request: PredictionRequest):
    model_type = request.model_type
    fasta = request.fasta

    # 驗證FASTA格式
    try:
        validate_fasta(fasta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 解析FASTA以獲取SMILES
    try:
        ids, smiles = parse_fasta(fasta)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse FASTA: {str(e)}")

    # 載入對應的模型
    try:
        model = load_model(model_type)
        print("Model loaded:", model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

    # 進行預測
    try:
        results = predict_toxicity(model, smiles)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # 構建回應
    response = {
        "status": "success",
        "data": {"fasta_ids": ids, "smiles": smiles, "predictions": results},
    }

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
