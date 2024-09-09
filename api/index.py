import requests, os
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from service.fpgnn.tool import get_scaler, load_args, load_data, load_model
from service.fpgnn.train import predict
from fastapi.responses import ORJSONResponse

app = FastAPI()


# 定義接收的 JSON 結構
# class PredictionRequest(BaseModel):
#     smiles: list[str]
#     model_type: str  # 用於指定 F2F, C2C 或 A2A 模型


# # 定義下載模型的函數
# def download_model_if_not_exists(url, model_name):
#     model_path = f"/tmp/{model_name}"  # 使用 /tmp 來保存下載的模型
#     if not os.path.exists(model_path):
#         print(f"Downloading model {model_name} from {url}...")
#         response = requests.get(url)
#         with open(model_path, "wb") as f:
#             f.write(response.content)
#         print(f"Model {model_name} downloaded.")
#     return model_path


# def predicting(smiles_list, model_path):
#     # 模擬一個從 SMILES 列表讀取的數據集，因為原始 load_data 需要文件
#     # 所以這裡可以創建一個臨時的 CSV 文件來傳遞數據
#     args = load_args(model_path)

#     # 使用 `load_data` 加載數據
#     test_data = load_data(smiles_list, args, from_file=False)

#     # 加載模型和進行預測
#     scaler = get_scaler(model_path)
#     model = load_model(model_path, args.cuda)

#     # 預測
#     test_pred = predict(model, test_data, args.batch_size, scaler)

#     # 將結果轉換為 Python 列表
#     test_pred = np.array(test_pred).tolist()

#     # 返回預測結果
#     return test_pred, test_data.smile()


@app.get("/api/python")
def hello_world():
    return {"message": "Hello World"}


# @app.post("/api/predict", response_class=ORJSONResponse)
# async def predict_toxicity(request: PredictionRequest):
#     model_map = {
#         "F2F": "https://bcoc8vpakuqoxkgl.public.blob.vercel-storage.com/F2F-BvHdyhk9tyo7d5onoB9iHpwiFoogAq.pt",
#         "C2C": "https://bcoc8vpakuqoxkgl.public.blob.vercel-storage.com/C2C-SpNj7ZlZAwHR6Nt4mrIkzEiUmLjhtQ.pt",
#         "A2A": "https://bcoc8vpakuqoxkgl.public.blob.vercel-storage.com/A2A-L8ecvXA89xTLYL1BP8gDvmepPtw4mY.pt",
#     }

#     if request.model_type not in model_map:
#         return {
#             "status": "false",
#             "message": "Invalid model_type. Please choose from F2F, C2C, A2A.",
#         }

#     # 根據 model_type 選擇模型的 Blob URL
#     model_url = model_map[request.model_type]
#     model_name = f"{request.model_type}.pt"

#     # 下載模型（如果尚未下載）
#     model_path = download_model_if_not_exists(model_url, model_name)

#     # 調用預測函數，傳入 SMILES 列表
#     # try:
#     #     predictions, smiles = predicting(request.smiles, model_path)
#     # except Exception as e:
#     #     return {"error": f"Prediction failed: {str(e)}"}

#     # 構建 JSON 格式的返回值
#     # result = {
#     #     "status": "success",
#     #     "data": {"smiles": smiles, "predictions": predictions},
#     # }

#     # return result
#     return {"message": "Hello World"}
