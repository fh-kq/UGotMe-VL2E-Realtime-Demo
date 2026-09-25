# -*- coding: utf-8 -*-
"""Local browser demo for the reproduced UGotMe/VL2E checkpoint."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys, json, tempfile
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from email.parser import BytesParser
from email.policy import default

ROOT = Path(__file__).resolve().parent
REPO = ROOT
sys.path.insert(0, str(REPO))
os.chdir(REPO)
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer
from models.vision_language_emotion import VLEModel
from datasets import get_transform

def square_crop(frame, box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * 0.7
    crop = frame[max(0, int(center_y - half)):min(height, int(center_y + half)),
                 max(0, int(center_x - half)):min(width, int(center_x + half))]
    return cv2.resize(crop, (160, 160)) if crop.size else None

LABELS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
LABEL_ZH = ["平静", "惊讶", "恐惧", "悲伤", "愉快", "厌恶", "愤怒"]
EMOJI = ["😐", "😮", "😨", "😢", "😊", "🤢", "😠"]
MAX_BODY = 48 * 1024 * 1024
MAX_FRAMES = 32

with initialize_config_dir(config_dir=str(REPO / "conf"), version_base=None):
    CFG = compose(config_name="config")
CFG.cwd = str(REPO)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = VLEModel(CFG).to(DEVICE)
CKPT = Path(os.environ.get("UGOTME_CHECKPOINT", str(ROOT / "results" / "trained_full" / "best.pth"))).expanduser()
if not CKPT.is_file():
    raise FileNotFoundError("找不到主模型权重: " + str(CKPT))
STATE = torch.load(CKPT, map_location="cpu", weights_only=True)
MODEL.load_state_dict(STATE["model"], strict=True)
MODEL.eval()
del STATE
TOKENIZER = AutoTokenizer.from_pretrained(CFG.model.text_encoder.pretrained_path, local_files_only=True)
DETECTOR = MTCNN(keep_all=True, device=DEVICE)
TRANSFORM = get_transform(CFG.data.transform, "val")

def text_tensor(history, utterance):
    def parse_turn(line):
        if ":" in line:
            who, words = line.split(":", 1)
            return who.strip() or "Speaker", words.strip()
        return "Speaker", line.strip()
    turns = [parse_turn(x) for x in history if x.strip()][-8:]
    speaker = "Speaker"
    encoded = [TOKENIZER(who + ":" + words)["input_ids"][1:] for who, words in turns]
    current = TOKENIZER(speaker + ":" + utterance.strip())["input_ids"][1:]
    seq = [TOKENIZER.cls_token_id]
    for item in encoded + [current]:
        seq.extend(item)
    seq.extend(TOKENIZER("For utterance:")["input_ids"][1:-1])
    seq.extend(current)
    seq.extend(TOKENIZER(speaker + " feels <mask>")["input_ids"][1:])
    seq = seq[-256:]
    seq += [TOKENIZER.pad_token_id] * (256 - len(seq))
    return torch.tensor([seq], dtype=torch.long, device=DEVICE)

def read_video_frames(video_bytes, suffix):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix or ".mp4", delete=False) as f:
            f.write(video_bytes); temp_path = f.name
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError("无法读取视频，请换用浏览器可解码的 MP4、AVI 或 MOV。")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError("视频中没有可读取的画面。")
        indices = np.unique(np.linspace(0, total - 1, min(total, MAX_FRAMES), dtype=int))
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok: frames.append(frame)
        cap.release()
        if not frames: raise ValueError("没有成功读取视频帧。")
        return frames
    finally:
        if temp_path and os.path.exists(temp_path): os.unlink(temp_path)

def prepare_vision(video_bytes, suffix):
    frames = read_video_frames(video_bytes, suffix)
    crops, held = [], None
    multi_count = detected_total = 0
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            boxes, probs = DETECTOR.detect(Image.fromarray(rgb))
        crop = None
        if boxes is not None and probs is not None:
            valid = probs >= 0.99
            boxes, probs = boxes[valid], probs[valid]
            detected_total += len(boxes)
            multi_count += int(len(boxes) > 1)
            if len(boxes):
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                crop = square_crop(frame, boxes[int(np.argmax(area))])
        if crop is None: crop = held
        if crop is not None:
            held = crop; crops.append(crop)
    if not crops:
        raise ValueError("视频里没有检测到清晰人脸，请上传正脸较清楚、光线充足的视频。")
    reference = crops[0]
    images = torch.zeros((1, 100, 3, 160, 160), dtype=torch.float32)
    mask = torch.zeros((1, 100), dtype=torch.float32)
    for i, crop in enumerate(crops[:100]):
        delta = np.subtract(crop, reference, dtype=np.uint8)
        images[0, i] = TRANSFORM(Image.fromarray(delta, mode="RGB"))
        mask[0, i] = 1
    return images.to(DEVICE), mask.to(DEVICE), {
        "sampled_frames": len(frames), "usable_face_frames": len(crops),
        "frames_with_multiple_faces": multi_count, "detected_faces": detected_total}

def predict(history, utterance, video_bytes, suffix):
    images, mask, info = prepare_vision(video_bytes, suffix)
    with torch.inference_mode():
        logits = MODEL(text_tensor(history, utterance), images, mask)[0].float()
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
    index = int(np.argmax(probs))
    return {"emotion": LABELS[index], "emotion_zh": LABEL_ZH[index], "emoji": EMOJI[index],
            "scores": [{"label": LABELS[i], "label_zh": LABEL_ZH[i],
                        "score": round(float(probs[i]) * 100, 2)} for i in range(7)],
            "info": info, "device": str(DEVICE), "checkpoint": CKPT.name}

PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UGotMe 情绪识别 Demo</title>
<style>
:root{--ink:#18233b;--muted:#6d7890;--blue:#4568f5;--line:#e6eaf2;--bg:#f5f7fb}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1060px;margin:36px auto;padding:0 20px}.head{display:flex;align-items:center;gap:14px;margin-bottom:22px}.logo{width:48px;height:48px;border-radius:16px;background:#e9edff;display:grid;place-items:center;font-size:26px}h1{font-size:23px;margin:0}.sub{color:var(--muted);margin-top:3px}.grid{display:grid;grid-template-columns:1.08fr .92fr;gap:18px}.card{background:white;border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 8px 30px #2030600a}h2{font-size:16px;margin:0 0 15px}label{display:block;font-weight:650;margin:15px 0 7px}textarea,input[type=file]{width:100%;border:1px solid #dfe4ee;border-radius:11px;padding:12px;font:inherit;background:white}textarea{resize:vertical;min-height:86px}.note{font-size:12px;color:var(--muted);margin-top:6px}.btn{margin-top:17px;border:0;border-radius:11px;padding:12px 18px;background:var(--blue);color:white;font-weight:700;font-size:15px;cursor:pointer}.btn:disabled{opacity:.55;cursor:wait}.result{min-height:430px;display:flex;flex-direction:column;align-items:center;justify-content:center}.robot{height:132px;width:132px;border-radius:38px;background:linear-gradient(145deg,#eef1ff,#dfe5ff);display:grid;place-items:center;font-size:74px}.emotion{font-size:25px;font-weight:750;margin-top:13px}.confidence{color:var(--muted);font-size:13px}.scores{width:100%;margin-top:24px}.row{display:grid;grid-template-columns:58px 1fr 48px;gap:10px;align-items:center;margin:9px 0;font-size:13px}.track{height:8px;background:#eef0f5;border-radius:9px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,#6d87ff,#4568f5);border-radius:9px}.status{margin-top:12px;min-height:22px;color:var(--muted);font-size:13px}.tag{display:inline-block;background:#f0f3fa;color:#58657b;border-radius:20px;padding:3px 9px;font-size:12px;margin:12px 5px 0 0}.foot{font-size:12px;color:#727e94;line-height:1.7;margin-top:18px}.hidden{display:none}@media(max-width:760px){.grid{grid-template-columns:1fr}main{margin:20px auto}.result{min-height:360px}}
</style></head><body><main><div class="head"><div class="logo">🤖</div><div><h1>UGotMe 情绪识别 Demo</h1><div class="sub">输入一段对话和说话人视频，查看 VL2E 的七类情绪判断</div></div></div>
<div class="grid"><section class="card"><h2>输入对话</h2><label for="history">前文（可选，每行一轮）</label><textarea id="history" placeholder="Alex: 今天的演讲怎么样？&#10;Speaker: 我有点紧张"></textarea><div class="note">用“说话人: 台词”格式，最多取最近 8 轮。</div>
<label for="utterance">当前说话内容</label><textarea id="utterance" placeholder="例如：我等了这么久，结果你告诉我取消了？"></textarea>
<label for="video">当前说话人的视频</label><input id="video" type="file" accept="video/*"><div class="note">建议 3–15 秒、正脸清楚的视频。</div><button class="btn" id="run">识别情绪</button><div class="status" id="status"></div>
<div class="foot">流程：视频抽帧 → MTCNN 检测人脸 → VL2E 融合人脸与文字 → 模拟机器人表情。此 demo 使用本地复现权重；多脸时选最大脸作近似定位，不使用声源方向。</div></section>
<section class="card result"><div class="robot" id="emoji">🤖</div><div class="emotion" id="emotion">等待输入</div><div class="confidence" id="confidence">识别结果将在这里显示</div><div class="scores hidden" id="scores"></div><div id="tags"></div></section></div></main>
<script>
var btn=document.getElementById("run");
btn.addEventListener("click",async function(){var utterance=document.getElementById("utterance").value.trim(),video=document.getElementById("video").files[0];if(!utterance){alert("请先填写当前说话内容");return}if(!video){alert("请先选择一段视频");return}
var form=new FormData();form.append("utterance",utterance);form.append("history",document.getElementById("history").value);form.append("video",video);btn.disabled=true;document.getElementById("status").textContent="正在抽帧、检测人脸并运行模型，请稍候…";
try{var response=await fetch("/predict",{method:"POST",body:form}),data=await response.json();if(!response.ok)throw new Error(data.error||"识别失败");document.getElementById("emoji").textContent=data.emoji;document.getElementById("emotion").textContent="识别结果："+data.emotion_zh;var top=data.scores.find(function(x){return x.label===data.emotion});document.getElementById("confidence").textContent="模型输出 "+top.score.toFixed(2)+"%（未经置信度校准）";
var box=document.getElementById("scores");box.classList.remove("hidden");box.innerHTML=data.scores.map(function(x){return '<div class="row"><span>'+x.label_zh+'</span><div class="track"><div class="fill" style="width:'+x.score+'%"></div></div><b>'+x.score.toFixed(1)+'%</b></div>'}).join("");
document.getElementById("tags").innerHTML='<span class="tag">可用人脸帧 '+data.info.usable_face_frames+'</span><span class="tag">多人脸帧 '+data.info.frames_with_multiple_faces+'</span><span class="tag">'+data.device+'</span>';document.getElementById("status").textContent="完成。权重："+data.checkpoint
}catch(e){document.getElementById("status").textContent="失败："+e.message}finally{btn.disabled=false}});
</script></body></html>"""

class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, code, payload, content_type):
        self.send_response(code); self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload))); self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers(); self.wfile.write(payload)
    def send_json(self, code, data):
        self.send_bytes(code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
    def do_GET(self):
        if self.path not in ("/", "/index.html"): return self.send_bytes(404, b"Not found", "text/plain")
        self.send_bytes(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
    def do_POST(self):
        if self.path != "/predict": return self.send_json(404, {"error":"Not found"})
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY: return self.send_json(413, {"error":"视频为空或超过48 MB。"})
        try:
            raw = self.rfile.read(length)
            head = ("Content-Type: " + self.headers.get("Content-Type", "") + "\r\nMIME-Version: 1.0\r\n\r\n").encode()
            msg = BytesParser(policy=default).parsebytes(head + raw)
            fields = {}
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if name: fields[name] = part.get_payload(decode=True) or b""
            utterance = fields.get("utterance", b"").decode("utf-8").strip()
            history = fields.get("history", b"").decode("utf-8").splitlines()
            video = fields.get("video", b"")
            suffix = Path(msg.get_filename() or "clip.mp4").suffix.lower()
            if not utterance or not video: raise ValueError("请提供当前台词和视频。")
            self.send_json(200, predict(history, utterance, video, suffix))
        except Exception as exc: self.send_json(400, {"error":str(exc)})
    def log_message(self, fmt, *args): print("[demo] " + fmt % args, flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run local UGotMe/VL2E demo.")
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("VL2E demo: http://%s:%d | device=%s" % (args.host, args.port, DEVICE), flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
