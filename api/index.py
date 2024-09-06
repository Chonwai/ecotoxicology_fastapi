from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from api.fpgnn.tool import get_scaler, load_args, load_data, load_model
from api.fpgnn.train import predict
from api.fpgnn.data import MoleDataSet
from typing import List
import io
import csv

app = FastAPI()


# 定義接收的 JSON 結構
class PredictionRequest(BaseModel):
    smiles: list[str]
    model_type: str  # 用於指定 F2F, C2C 或 A2A 模型


# 定義返回的 JSON 結構
class PredictionResponse(BaseModel):
    smiles: List[str]
    predictions: List[float]


# def predicting(smiles_list: list, model_path: str):
#     # 模擬一個從 SMILES 列表讀取的 DataLoader, 假設 MoleDataSet 支持從 SMILES 列表初始化
#     test_data = MoleDataSet(smiles_list)
    
#     # 加載模型
#     scaler = get_scaler(model_path)
#     train_args = load_args(model_path)
#     model = load_model(model_path, train_args.cuda)
    
#     # 進行預測
#     test_pred = predict(model, test_data, train_args.batch_size, scaler)
    
#     # 檢查預測結果並展開為單層結構
#     test_pred = np.array(test_pred).flatten().tolist()  # 確保為單層的浮點數列表
#     return test_pred

def predicting(smiles_list, model_path):
    # 模擬一個從 SMILES 列表讀取的數據集，因為原始 load_data 需要文件
    # 所以這裡可以創建一個臨時的 CSV 文件來傳遞數據
    args = load_args(model_path)

    # 使用 `load_data` 加載數據
    test_data = load_data(smiles_list, args, from_file=False)
    
    # 加載模型和進行預測
    scaler = get_scaler(model_path)
    model = load_model(model_path, args.cuda)
    
    # 預測
    test_pred = predict(model, test_data, args.batch_size, scaler)
    
    # 將結果轉換為 Python 列表
    test_pred = np.array(test_pred).tolist()
    
    # 返回預測結果
    return test_pred, test_data.smile()


@app.get("/api/python")
def hello_world():
    return {"message": "Hello World"}


"""
分子毒性预测接口，接受 SMILES 列表，并返回对应的预测值。

:param request: 包含 SMILES 列表和模型类型的 JSON 请求。
:return: 返回预测结果的 JSON 格式。
"""
@app.post("/api/predict")
def predict_toxicity(request: PredictionRequest):
    # 根據 model_type 選擇對應的模型路徑
    model_map = {"F2F": "api/model/F2F.pt", "C2C": "api/model/C2C.pt", "A2A": "api/model/A2A.pt"}

    if request.model_type not in model_map:
        return {"status": "false", "message": "Invalid model_type. Please choose from F2F, C2C, A2A."}
    
    print(request.smiles)
    print(request.model_type)

    model_path = model_map[request.model_type]
    
    # 調用預測函數，傳入 SMILES 列表
    try:
        predictions, smiles = predicting(request.smiles, model_path)
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # 構建 JSON 格式的返回值
    result = {
        "status": "success",
        "data": {
            "smiles": smiles,
            "predictions": predictions
        }
    }

    return result
