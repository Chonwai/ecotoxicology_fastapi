from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from service.fpgnn.tool import get_scaler, load_args, load_data, load_model
from service.fpgnn.train import predict

app = FastAPI()


# 定義接收的 JSON 結構
class PredictionRequest(BaseModel):
    fasta: str
    model_type: str  # 用於指定 F2F, C2C 或 A2A 模型


def validate_fasta(fasta_str):
    # 清理換行符
    fasta_str = fasta_str.replace('\r\n', '\n').replace('\r', '\n')
    lines = fasta_str.strip().split('\n')
    if len(lines) % 2 != 0:
        raise ValueError("FASTA format error: Each ID line must be followed by a sequence line.")
    for i in range(0, len(lines), 2):
        if not lines[i].startswith('>'):
            raise ValueError(f"FASTA format error: Line {i+1} does not start with '>'.")
        if not lines[i+1]:
            raise ValueError(f"FASTA format error: Line {i+2} is empty.")
    return True


def parse_fasta(fasta_str):
    # 清理換行符
    fasta_str = fasta_str.replace('\r\n', '\n').replace('\r', '\n')
    lines = fasta_str.strip().split('\n')
    ids = []
    smiles = []
    for i in range(0, len(lines), 2):
        ids.append(lines[i][1:])  # 去掉 '>'
        smiles.append(lines[i+1])
    return ids, smiles


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
    
    # 將結果轉換為一維數組
    test_pred = np.array(test_pred).flatten().tolist()
    
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
    model_map = {"F2F": "service/model/F2F.pt", "C2C": "service/model/C2C.pt", "A2A": "service/model/A2A.pt"}

    if request.model_type not in model_map:
        return {"status": "false", "message": "Invalid model_type. Please choose from F2F, C2C, A2A."}
    
    print(request)
    
    try:
        validate_fasta(request.fasta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    ids, smiles = parse_fasta(request.fasta)
    model_path = model_map[request.model_type]
    
    # 調用預測函數，傳入 SMILES 列表
    try:
        predictions, _ = predicting(smiles, model_path)
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    # 構建 JSON 格式的返回值
    result = {
        "status": "success",
        "data": {
            "fasta_ids": ids,
            "smiles": smiles,
            "predictions": predictions
        }
    }
    
    print(result)

    return result

if __name__ == "__main__":
    app.run(debug=True)