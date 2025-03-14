import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch_geometric.datasets import Planetoid
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import kiến trúc mô hình GCN
from model.gcn import GCN

# 🔹 Load dataset (Cora)
dataset = Planetoid(root="data", name="Cora")

# Lấy số feature và số lớp từ dataset
num_features = dataset.num_node_features
num_classes = dataset.num_classes

# Danh sách tên class tương ứng với tập Cora
class_names = [
    "Case-Based",               # Class 0
    "Genetic Algorithms",       # Class 1
    "Neural Networks",          # Class 2
    "Probabilistic Methods",    # Class 3
    "Reinforcement Learning",   # Class 4
    "Rule Learning",            # Class 5
    "Theory"                    # Class 6
]

# 🔹 Khởi tạo FastAPI
app = FastAPI()

# 🔹 Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Tải mô hình đã huấn luyện
device = torch.device("cpu")
model_path = os.path.join(os.path.dirname(__file__), "../model/gcn_model.pth")

# 🛠 Khởi tạo model với số feature và số class đúng
model = GCN(num_features=num_features, hidden_channels=16, num_classes=num_classes)

# 🛠 Load trọng số
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 🔹 Lớp nhận dữ liệu đầu vào
class PaperInput(BaseModel):
    title: str

# 🔹 Dữ liệu đồ thị từ tập Cora
data = dataset[0]
features = data.x.clone().to(device)  # Sao chép đặc trưng gốc
edge_index = data.edge_index.clone().to(device)  # Sao chép đồ thị gốc

@app.post("/predict")
async def predict_paper(paper: PaperInput):
    global features, edge_index  # Cập nhật global để thêm nút mới

    title = paper.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Vui lòng cung cấp tiêu đề bài báo!")

    # 🛠 Tạo vector đặc trưng ngẫu nhiên (thay thế bằng embedding thực tế nếu có)
    new_feature = torch.randn(1, num_features).to(device)  # (1, d)

    # 🛠 Thêm vào danh sách đặc trưng
    N = features.shape[0]  # Số nút hiện có
    features = torch.cat([features, new_feature], dim=0)  # (N+1, d)

    # 🛠 Kết nối với nút gần nhất bằng cosine similarity
    similarities = torch.cosine_similarity(new_feature, features[:-1], dim=1)
    existing_node_index = torch.argmax(similarities).item()

    # 🛠 Cập nhật tập cạnh `edge_index`
    new_edge = torch.tensor([[existing_node_index], [N]], dtype=torch.long, device=device)
    edge_index = torch.cat([edge_index, new_edge], dim=1)  # Thêm liên kết mới

    # 🔹 Dự đoán với mô hình GCN
    with torch.no_grad():
        output = model(features, edge_index)
        if N >= output.shape[0]:  # Kiểm tra lỗi chỉ mục
            raise HTTPException(status_code=500, detail="Lỗi: chỉ số nút vượt quá phạm vi.")

        probabilities = F.softmax(output[N], dim=0)  # Lấy xác suất của nút mới

    # 🔹 Kiểm tra kết quả dự đoán
    if probabilities.numel() != num_classes:
        raise HTTPException(status_code=500, detail="Lỗi: Số lượng lớp dự đoán không khớp.")

    # 🔹 Lấy top 3 dự đoán
    top_probs, top_classes = torch.topk(probabilities, min(3, num_classes))
    predictions = [
        {
            "class": class_names[int(c.item())],  # Gán tên class thay vì số
            "confidence": round(float(p.item()) * 100, 2)
        }
        for p, c in zip(top_probs, top_classes)
    ]

    return {"title": title, "predictions": predictions}