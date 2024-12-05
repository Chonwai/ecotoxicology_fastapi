import torch
from torch_geometric.data import Data
import pickle
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from service.GCN import Smile2Graph

sys.path.append('.')

class SavedGCN(torch.nn.Module):
    def __init__(self):
        super(SavedGCN, self).__init__()
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.lin = None

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        return x

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 啟動時執行
    # import sys
    # sys.modules['__main__'].SavedGCN = SavedGCN
    
    for model_name, path in MODEL_PATHS.items():
        try:
            print(f"嘗試載入模型: {path}")
            with open(path, 'rb') as f:
                models[model_name] = pickle.load(f)
            print(f"成功載入模型: {model_name}")
        except Exception as e:
            print(f"無法載入模型 {model_name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
    yield
    # 關閉時執行
    models.clear()

app = FastAPI(
    title="生態毒理學預測API",
    description="用於預測化合物對魚類、甲殼類和藻類的毒性",
    version="1.0.0",
    lifespan=lifespan
)

class PredictRequest(BaseModel):
    fasta: str
    model_type: str

class PredictData(BaseModel):
    fasta_ids: List[str]
    smiles: List[str]
    predictions: List[float]

class PredictResponse(BaseModel):
    status: str
    data: PredictData

MODEL_PATHS = {
    "F2F": "service/model/F2F.pickle",
    "C2C": "service/model/C2C.pickle",
    "A2A": "service/model/A2A.pickle"
}

models = {}

def parse_fasta(fasta_str: str) -> List[str]:
    lines = fasta_str.strip().split('\n')
    smiles_list = []
    current_smiles = ""
    
    for line in lines:
        if line.startswith('>'):
            if current_smiles:
                smiles_list.append(current_smiles)
            current_smiles = ""
        else:
            current_smiles += line
    
    if current_smiles:
        smiles_list.append(current_smiles)
        
    return smiles_list

@app.post("/api/predict")
async def predict(request: PredictRequest):
    if request.model_type not in models:
        raise HTTPException(status_code=400, detail="不支持的模型類型")
    
    model = models[request.model_type]
    if not model:
        raise HTTPException(status_code=500, detail="模型未正確載入")
    
    fasta_lines = request.fasta.strip().split('\n')
    fasta_ids = []
    smiles_list = []
    current_id = ""
    current_smiles = ""
    
    for line in fasta_lines:
        if line.startswith('>'):
            if current_smiles:
                smiles_list.append(current_smiles)
            current_id = line[1:].strip()  # 移除 '>' 並去除空白
            fasta_ids.append(current_id)
            current_smiles = ""
        else:
            current_smiles += line
    
    if current_smiles:
        smiles_list.append(current_smiles)
    
    predictions = []
    for smi in smiles_list:
        try:
            g = Smile2Graph(smi, label=0)
            data = Data(
                x=g.get_node_feature(),
                edge_index=g.get_edge_index().t().contiguous(),
                edge_attr=g.get_edge_feature(),
                y=g.get_y()
            )
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.softmax(out, dim=1)
            predictions.append(probs[0, 1].float().item())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"處理SMILES時出錯: {str(e)}")
    
    return PredictResponse(
        status="success",
        data=PredictData(
            fasta_ids=fasta_ids,
            smiles=smiles_list,
            predictions=predictions
        )
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
