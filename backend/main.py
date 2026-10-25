import io
import os
import time
import uuid
import cv2
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google import genai
from lime import lime_image
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import yt_dlp

# .env 파일 로드
load_dotenv()

app = FastAPI()

# ---------------------------------------------------------
# 0. Gemini 설정 및 세션 관리
# ---------------------------------------------------------
GENAI_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GENAI_KEY, http_options={"api_version": "v1"})
MODEL_NAME = "gemini-2.5-flash"

chat_sessions = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def cleanup_old_files(directory: str, max_age_seconds: int = 3600):
    try:
        now = time.time()
        for filename in os.listdir(directory):
            if filename.startswith("lime_"):
                file_path = os.path.join(directory, filename)
                if os.path.getmtime(file_path) < now - max_age_seconds:
                    os.remove(file_path)
    except Exception as e:
        print(f"⚠️ Cleanup Error: {e}")


# ---------------------------------------------------------
# 2. MesoNet2 모델 정의
# ---------------------------------------------------------
class MesoNet2(nn.Module):
    def __init__(self, num_classes=2):
        super(MesoNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.maxpool(x)
        x = self.adaptive_pool(x)
        return self.fc(x.view(x.size(0), -1))


device = torch.device("cpu")  # GPU 사용 시 "cuda"로 변경
eff_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
meso_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

try:
    eff_model = models.efficientnet_b0(weights=None)
    eff_model.classifier[1] = nn.Linear(eff_model.classifier[1].in_features, 2)
    eff_model.load_state_dict(
        torch.load("models/EfficientNet.pth", map_location=device)
    )
    eff_model.eval()
    meso_model = MesoNet2().to(device)
    meso_model.load_state_dict(torch.load("models/MesoNet.pth", map_location=device))
    meso_model.eval()
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Model Load Error: {e}")

explainer = lime_image.LimeImageExplainer()


def batch_predict_eff(imgs):
    imgs_torch = torch.stack(
        [eff_transform(Image.fromarray((i * 255).astype(np.uint8))) for i in imgs]
    ).to(device)
    return F.softmax(eff_model(imgs_torch), dim=1).detach().cpu().numpy()


def batch_predict_meso(imgs):
    imgs_torch = torch.stack(
        [meso_transform(Image.fromarray((i * 255).astype(np.uint8))) for i in imgs]
    ).to(device)
    return F.softmax(meso_model(imgs_torch), dim=1).detach().cpu().numpy()


# ---------------------------------------------------------
# ✅ 1. 공학적 근거 기반 리포트 생성기
# ---------------------------------------------------------
async def get_llm_report(data: dict):
    eff_result = "FAKE" if data["eff_score"] >= 0.5 else "REAL"
    meso_result = "FAKE" if data["meso_score"] >= 0.5 else "REAL"

    if data["is_discrepancy"]:
        consensus_status = f"Discrepancy detected: Primary model predicted {eff_result}, auxiliary model predicted {meso_result}."
    else:
        consensus_status = (
            f"Consensus achieved: Both models classified the media as {eff_result}."
        )

    prompt = f"""
    You are an expert forensic analyst for the deepfake detection system 'DeepEye'.
    Write a concise forensic analysis report within 10 lines in fluent English based on the following engineering evidence and empirical observations.
    All outputs must strictly be written in English.

    [Key Data]
    - Final Verdict: {data['result']} (Confidence: {data['final_score'] * 100:.2f}%)
    - Primary Model (EffNet) Score: {data['eff_score']:.4f} ({eff_result})
    - Auxiliary Model (MesoNet) Score: {data['meso_score']:.4f} ({meso_result})
    - Agreement Status: {consensus_status}

    [Guidelines]
    1. Technical & Empirical Explanation: Refer to fine-grained visual artifacts and link visual cues with phrases such as "Notably, unnatural noise clustering around [facial region] highlighted in the heatmap is clearly observed."
    2. Data Fidelity: You must explicitly cite the scores and metrics provided above in the report.
    3. Engineering Perspectives: Explain EffNet's findings in terms of 'skin texture and noise patterns', and MesoNet's findings in terms of 'mesoscopic facial structure and lighting consistency'.
    4. Tone & Format: Professional, objective forensic tone. Output pure plain text without markdown formatting (do not use asterisks or bold text).
    5. Closing Statement: The final sentence must strictly be: "Up to 3 follow-up questions are allowed."
    """

    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ [Gemini API Error Detail]: {e}")
        return "An error occurred while generating the forensic analysis report."


# ---------------------------------------------------------
# ✅ 3. 분석 메인 로직 (타이머 측정 포함)
# ---------------------------------------------------------
async def analyze_image(image: Image.Image, unique_id: str):
    inference_start = time.time()

    image_rgb = image.convert("RGB").resize((224, 224))
    img_array = np.array(image_rgb) / 255.0

    with torch.no_grad():
        eff_in = eff_transform(image_rgb).unsqueeze(0).to(device)
        eff_prob = F.softmax(eff_model(eff_in), dim=1)[0][1].item()
        meso_in = meso_transform(image_rgb).unsqueeze(0).to(device)
        meso_prob = F.softmax(meso_model(meso_in), dim=1)[0][1].item()

    CUSTOM_THRESHOLD = 0.4
    eff_result = "FAKE" if eff_prob > CUSTOM_THRESHOLD else "REAL"
    meso_result = "FAKE" if meso_prob > CUSTOM_THRESHOLD else "REAL"

    exp_eff = explainer.explain_instance(
        img_array, batch_predict_eff, top_labels=1, hide_color=0, num_samples=250
    )
    _, mask_eff = exp_eff.get_image_and_mask(
        exp_eff.top_labels[0],
        positive_only=True,
        num_features=3,
        hide_rest=False,
        min_weight=0.05,
    )

    overlay = np.zeros((224, 224, 3), dtype=np.uint8)

    if eff_result == meso_result:
        exp_meso = explainer.explain_instance(
            img_array, batch_predict_meso, top_labels=1, hide_color=0, num_samples=250
        )
        _, mask_meso = exp_meso.get_image_and_mask(
            exp_meso.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False,
            min_weight=0.05,
        )

        for y in range(224):
            for x in range(224):
                if mask_eff[y, x] and mask_meso[y, x]:
                    overlay[y, x] = [255, 0, 0]
                elif mask_eff[y, x]:
                    overlay[y, x] = [255, 255, 0]
                elif mask_meso[y, x]:
                    overlay[y, x] = [255, 165, 0]
    else:
        for y in range(224):
            for x in range(224):
                if mask_eff[y, x]:
                    overlay[y, x] = [255, 255, 0]

    combined = cv2.addWeighted(np.array(image_rgb), 0.5, overlay, 0.5, 0)
    lime_image_name = f"lime_{unique_id}.png"
    Image.fromarray(combined).save(os.path.join(STATIC_DIR, lime_image_name))

    adjusted_score = min(eff_prob, 0.9992) if eff_prob > 0.5 else max(eff_prob, 0.0008)

    analysis_result = {
        "eff_score": round(eff_prob, 4),
        "meso_score": round(meso_prob, 4),
        "final_score": round(adjusted_score, 4),
        "result": eff_result,
        "lime_url": f"http://127.0.0.1:8000/static/{lime_image_name}",
        "is_discrepancy": eff_result != meso_result,
        "session_id": unique_id,
    }

    inference_end = time.time()
    step2_time = inference_end - inference_start

    llm_start = time.time()
    analysis_result["llm_report"] = await get_llm_report(analysis_result)
    llm_end = time.time()
    step3_time = llm_end - llm_start

    print(f"\n[Performance Measurement - ID: {unique_id}]")
    print(f"Phase 2 (Inference & LIME): {step2_time:.4f}s")
    print(f"Phase 3 (Gemini Report): {step3_time:.4f}s")

    analysis_result["step2_time"] = step2_time
    analysis_result["step3_time"] = step3_time

    return analysis_result


# ---------------------------------------------------------
# ✅ API Endpoints (1단계 측정 포함)
# ---------------------------------------------------------
@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    step1_start = time.time()
    background_tasks.add_task(cleanup_old_files, STATIC_DIR)
    try:
        contents = await file.read()
        if file.content_type.startswith("video/") or file.filename.lower().endswith(
            (".mp4", ".avi", ".mov")
        ):
            temp_name = f"temp_{uuid.uuid4()}.mp4"
            with open(temp_name, "wb") as f:
                f.write(contents)
            cap = cv2.VideoCapture(temp_name)
            _, frame = cap.read()
            cap.release()
            if os.path.exists(temp_name):
                os.remove(temp_name)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            image = Image.open(io.BytesIO(contents))

        step1_time = time.time() - step1_start
        unique_id = str(uuid.uuid4())
        chat_sessions[unique_id] = 0

        result = await analyze_image(image, unique_id)
        print(f"Phase 1 (Local Loading): {step1_time:.4f}s")
        print(
            f"TOTAL TIME: {step1_time + result['step2_time'] + result['step3_time']:.4f}s"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-link")
async def predict_link(background_tasks: BackgroundTasks, payload: dict = Body(...)):
    step1_start = time.time()
    background_tasks.add_task(cleanup_old_files, STATIC_DIR)
    try:
        url = payload.get("url")
        start_time = payload.get("startTime", 0)
        ydl_opts = {"format": "best", "quiet": True, "noplaylist": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_url = info["url"]
        cap = cv2.VideoCapture(video_url)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        _, frame = cap.read()
        cap.release()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        step1_time = time.time() - step1_start
        unique_id = str(uuid.uuid4())
        chat_sessions[unique_id] = 0

        result = await analyze_image(image, unique_id)
        print(f"Phase 1 (SNS Loading): {step1_time:.4f}s")
        print(
            f"TOTAL TIME: {step1_time + result['step2_time'] + result['step3_time']:.4f}s"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_with_deepeye(payload: dict = Body(...)):
    user_msg = payload.get("message")
    context = payload.get("context", "No context available")
    lime_url = payload.get("lime_url", "")
    session_id = payload.get("session_id", "default_session")
    current_count = chat_sessions.get(session_id, 0)
    if current_count >= 3:
        return {
            "reply": "You have exceeded the maximum limit of 3 follow-up questions."
        }
    try:
        content_list = [
            f"You are the DeepEye visual forensic AI. Analyze the LIME attribution image and the context, then answer the user question strictly in fluent English as plain text (no markdown).\nContext: {context}\nQuestion: {user_msg}"
        ]
        if lime_url:
            img_path = os.path.join(STATIC_DIR, lime_url.split("/")[-1])
            if os.path.exists(img_path):
                content_list.append(Image.open(img_path))
        response = client.models.generate_content(
            model=MODEL_NAME, contents=content_list
        )
        chat_sessions[session_id] = current_count + 1
        return {"reply": response.text.strip()}
    except Exception as e:
        return {"reply": "An error occurred while generating the response."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
