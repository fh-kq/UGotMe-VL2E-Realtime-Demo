# -*- coding: utf-8 -*-
"""Real-time webcam demo using the local UGotMe/VL2E reproduction."""
import base64
import json
from http.server import ThreadingHTTPServer
from email import policy
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import vl2e_demo as core
LIVE_REFERENCE = None

PAGE = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UGotMe 实时情绪监测</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fb;color:#18233b;font:15px/1.55 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1100px;margin:30px auto;padding:0 18px}.head{margin-bottom:20px}h1{margin:0;font-size:24px}.sub,.note{color:#6d7890}.sub{margin-top:4px}.grid{display:grid;grid-template-columns:1.05fr .95fr;gap:18px}.card{background:white;border:1px solid #e5e9f1;border-radius:18px;padding:20px;box-shadow:0 8px 30px #2030600a}h2{font-size:16px;margin:0 0 12px}label{display:block;font-weight:650;margin:12px 0 6px}textarea{width:100%;border:1px solid #dfe4ee;border-radius:10px;padding:11px;font:inherit;resize:vertical;min-height:66px}select{width:100%;border:1px solid #dfe4ee;border-radius:10px;padding:10px;font:inherit;background:white}video{display:block;width:100%;height:360px;background:#121827;border-radius:12px;object-fit:contain}#sampled{width:128px;height:96px;object-fit:contain;border-radius:8px;background:#121827}.samplebar{display:flex;align-items:center;gap:10px;margin-top:8px;color:#6d7890;font-size:12px}button{border:0;border-radius:10px;padding:11px 16px;margin:12px 8px 0 0;background:#4568f5;color:white;font-weight:700;cursor:pointer}button.stop{background:#e9edf6;color:#33415d}button:disabled{opacity:.5;cursor:wait}.note{font-size:12px;margin-top:6px}.result{display:flex;min-height:450px;flex-direction:column;align-items:center;justify-content:center}.face{width:130px;height:130px;border-radius:36px;background:#e9edff;display:grid;place-items:center;font-size:75px}.emotion{font-size:25px;font-weight:750;margin-top:12px}.hint{font-size:13px;color:#6d7890}.scores{width:100%;margin-top:22px}.row{display:grid;grid-template-columns:55px 1fr 48px;gap:10px;align-items:center;margin:9px 0;font-size:13px}.track{height:8px;border-radius:8px;background:#eef0f5;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,#7990ff,#4568f5)}.status{font-size:13px;color:#68758c;margin-top:10px;min-height:20px}.tag{font-size:12px;color:#53617a;background:#f0f3fa;padding:4px 9px;border-radius:20px;margin:10px 4px}.hidden{display:none}.foot{color:#6d7890;font-size:12px;line-height:1.7;margin-top:16px}@media(max-width:760px){.grid{grid-template-columns:1fr}main{margin:18px auto}.result{min-height:370px}}
</style></head><body><main><div class="head"><h1>UGotMe 实时情绪监测</h1><div class="sub">本地摄像头连续取样，VL2E 约每 3 秒更新一次情绪判断</div></div>
<div class="grid"><section class="card"><h2>摄像头与对话</h2><label for="cameraSelect">选择摄像头</label><select id="cameraSelect"><option value="">默认摄像头</option></select><div class="note">先开启一次摄像头以读取设备名称；若画面是“超级互联”图标，请在这里改选“Integrated Camera / 内置摄像头”。</div><video id="cam" autoplay muted playsinline></video><div id="caminfo" class="note">请开启摄像头，并在浏览器弹出的权限框中允许访问。</div><div class="samplebar">送入模型的最新画面：<img id="sampled" alt="等待摄像头画面"></div>
<div><button id="start">开启摄像头</button><button id="switchCamera" class="stop" disabled>切换到所选摄像头</button><button id="stop" class="stop" disabled>停止监测</button></div>
<label for="history">对话前文（可选，每行一轮）</label><textarea id="history" placeholder="Alex: 今天怎么样？"></textarea>
<label for="utterance">当前说话内容 / 转写文本</label><textarea id="utterance" placeholder="输入当前说话内容；说话过程中可随时修改"></textarea>
<div class="note">模型每次使用最新的摄像头画面和当前文本。请将说话人保持在画面中；多人时暂以最大人脸近似选择说话人。</div>
<div id="status" class="status">尚未开启摄像头</div>
<div class="foot">画面只发送到本机运行的模型服务。这个模拟版没有接麦克风、自动语音转写、声源定位或机器人转头；分类每约 3 秒刷新一次，接近论文的在线对话演示流程。</div></section>
<section class="card result"><div class="face" id="emoji">🤖</div><div class="emotion" id="emotion">等待监测</div><div class="hint" id="confidence">识别分布将在这里显示</div><div id="scores" class="scores hidden"></div><div id="tags"></div></section></div></main>
<script>
var active=false,stream=null,samples=[],busy=false,timer=null;
var cam=document.getElementById("cam"),start=document.getElementById("start"),stop=document.getElementById("stop");
var cameraSelect=document.getElementById("cameraSelect"),switchCamera=document.getElementById("switchCamera");
function wait(ms){return new Promise(function(ok){setTimeout(ok,ms)})}
async function refreshCameras(){var current="";if(stream&&stream.getVideoTracks().length)current=stream.getVideoTracks()[0].getSettings().deviceId||"";var devices=(await navigator.mediaDevices.enumerateDevices()).filter(function(d){return d.kind==="videoinput"});cameraSelect.innerHTML="";devices.forEach(function(d,i){var o=document.createElement("option");o.value=d.deviceId;o.textContent=d.label||("摄像头 "+(i+1));cameraSelect.appendChild(o)});if(current)cameraSelect.value=current;switchCamera.disabled=!active||devices.length<2}
async function connectCamera(){if(stream)stream.getTracks().forEach(function(t){t.stop()});var selected=cameraSelect.value;var videoSpec=selected?{deviceId:{exact:selected},width:{ideal:640},height:{ideal:480}}:{facingMode:"user",width:{ideal:640},height:{ideal:480}};stream=await navigator.mediaDevices.getUserMedia({video:videoSpec,audio:false});cam.srcObject=stream;await cam.play();var settings=stream.getVideoTracks()[0].getSettings();document.getElementById("caminfo").textContent="当前摄像头："+(settings.width||cam.videoWidth)+"×"+(settings.height||cam.videoHeight)+"；请确认预览里能看到你的脸。";await refreshCameras()}
start.onclick=async function(){try{await fetch("/reset",{method:"POST"});await connectCamera();active=true;start.disabled=true;stop.disabled=false;switchCamera.disabled=false;document.getElementById("status").textContent="摄像头已开启，正在收集画面…";samples=[];timer=setInterval(function(){if(!active||!cam.videoWidth)return;var c=document.createElement("canvas");c.width=cam.videoWidth;c.height=cam.videoHeight;c.getContext("2d").drawImage(cam,0,0);var shot=c.toDataURL("image/jpeg",0.72);document.getElementById("sampled").src=shot;samples.push(shot.split(",")[1]);if(samples.length>8)samples.shift()},500);monitor()}catch(e){document.getElementById("status").textContent="无法开启摄像头："+e.message}};
switchCamera.onclick=async function(){try{await connectCamera();await fetch("/reset",{method:"POST"});samples=[];document.getElementById("status").textContent="已切换摄像头，正在重新取样…"}catch(e){document.getElementById("status").textContent="切换失败："+e.message}};
stop.onclick=function(){active=false;clearInterval(timer);if(stream)stream.getTracks().forEach(function(t){t.stop()});cam.srcObject=null;start.disabled=false;stop.disabled=true;switchCamera.disabled=true;document.getElementById("status").textContent="监测已停止"};async function monitor(){while(active){await wait(3000);if(!active||busy||samples.length<4)continue;busy=true;document.getElementById("status").textContent="正在分析最近几秒的画面…";try{var r=await fetch("/live_predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({frames:samples.slice(-8),history:document.getElementById("history").value,utterance:document.getElementById("utterance").value})});var d=await r.json();if(!r.ok)throw new Error(d.error||"推理失败");document.getElementById("emoji").textContent=d.emoji;document.getElementById("emotion").textContent="当前判断："+d.emotion_zh;var top=d.scores.find(function(x){return x.label===d.emotion});document.getElementById("confidence").textContent="模型输出 "+top.score.toFixed(2)+"%（未经校准）";var box=document.getElementById("scores");box.classList.remove("hidden");box.innerHTML=d.scores.map(function(x){return '<div class="row"><span>'+x.label_zh+'</span><div class="track"><div class="fill" style="width:'+x.score+'%"></div></div><b>'+x.score.toFixed(1)+'%</b></div>'}).join("");document.getElementById("tags").innerHTML='<span class="tag">人脸帧 '+d.info.usable_face_frames+'</span><span class="tag">'+d.device+'</span>';document.getElementById("tags").innerHTML += '<span class="tag">本次真正检测到人脸 '+d.info.fresh_face_frames+'/'+d.info.usable_face_frames+' 帧</span><span class="tag">与初始人脸的平均像素差 '+d.info.face_change+'/255（不是情绪强度）</span>';document.getElementById("status").textContent="监测中 · "+new Date().toLocaleTimeString()}catch(e){document.getElementById("status").textContent="分析提示："+e.message}finally{busy=false}}}
</script></body></html>"""

def predict_frames(encoded_frames, history, utterance):
    crops, held = [], None
    multi = 0
    max_confidence = 0.0
    fresh_faces = 0
    for item in encoded_frames:
        raw = base64.b64decode(item, validate=True)
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            boxes, probs = core.DETECTOR.detect(Image.fromarray(rgb))
        crop = None
        if boxes is not None and probs is not None:
            if len(probs):
                max_confidence = max(max_confidence, float(np.max(probs)))
            keep = probs >= 0.90
            if len(probs) and not keep.any() and float(np.max(probs)) >= 0.70:
                keep[int(np.argmax(probs))] = True
            boxes, probs = boxes[keep], probs[keep]
            multi += int(len(boxes) > 1)
            if len(boxes):
                areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
                crop = core.square_crop(frame, boxes[int(np.argmax(areas))])
                if crop is not None:
                    fresh_faces += 1
        if crop is None:
            crop = held
        if crop is not None:
            held = crop
            crops.append(crop)
    if not crops:
        raise ValueError("未检测到可用人脸；最高检测分 %.2f。请正对摄像头、靠近一些并改善光线。" % max_confidence)
    global LIVE_REFERENCE
    if LIVE_REFERENCE is None:
        LIVE_REFERENCE = crops[0].copy()
    ref = LIVE_REFERENCE
    face_change = float(np.mean([
        np.mean(np.abs(crop.astype(np.int16) - ref.astype(np.int16)))
        for crop in crops
    ]))
    images = torch.zeros((1,100,3,160,160), dtype=torch.float32)
    mask = torch.zeros((1,100), dtype=torch.float32)
    for i, crop in enumerate(crops[:100]):
        delta = np.subtract(crop, ref, dtype=np.uint8)
        images[0,i] = core.TRANSFORM(Image.fromarray(delta, mode="RGB"))
        mask[0,i] = 1
    with torch.inference_mode():
        logits = core.MODEL(core.text_tensor(history.splitlines(), utterance or " "), images.to(core.DEVICE), mask.to(core.DEVICE))[0].float()
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
    k = int(np.argmax(probs))
    return {"emotion":core.LABELS[k],"emotion_zh":core.LABEL_ZH[k],"emoji":core.EMOJI[k],
            "scores":[{"label":core.LABELS[i],"label_zh":core.LABEL_ZH[i],"score":round(float(probs[i])*100,2)} for i in range(7)],
            "info":{"usable_face_frames":len(crops),"fresh_face_frames":fresh_faces,
                    "face_change":round(face_change,1),"frames_with_multiple_faces":multi},
            "device":str(core.DEVICE)}

class LiveHandler(core.Handler):
    def do_GET(self):
        if self.path not in ("/","/index.html"):
            return self.send_bytes(404,b"Not found","text/plain")
        self.send_bytes(200,PAGE.encode("utf-8"),"text/html; charset=utf-8")
    def do_POST(self):
        global LIVE_REFERENCE
        if self.path == "/reset":
            LIVE_REFERENCE = None
            return self.send_json(200,{"ok":True})
        if self.path != "/live_predict":
            return self.send_json(404,{"error":"Not found"})
        length = int(self.headers.get("Content-Length","0"))
        if length <= 0 or length > 12*1024*1024:
            return self.send_json(413,{"error":"画面数据为空或过大。"})
        try:
            data=json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data.get("frames"),list) or len(data["frames"]) < 1:
                raise ValueError("还没有摄像头画面。")
            result=predict_frames(data["frames"][-8:],str(data.get("history","")),str(data.get("utterance","")))
            self.send_json(200,result)
        except Exception as exc:
            self.send_json(400,{"error":str(exc)})

if __name__ == "__main__":
    server=ThreadingHTTPServer(("127.0.0.1",8766),LiveHandler)
    print("实时 Demo: http://127.0.0.1:8766",flush=True)
    print("模型设备: "+str(core.DEVICE)+"；退出请按 Ctrl+C。",flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
