import os
import sys
import json
import base64
import io
import time
import argparse
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import Counter
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

ov_engine = None

COCO_JA_MAP = {
    'person': '人物', 'bicycle': '自転車', 'car': '自動車', 'motorcycle': 'バイク',
    'airplane': '飛行機', 'bus': 'バス', 'train': '電車', 'truck': 'トラック',
    'boat': '船', 'traffic light': '信号', 'fire hydrant': '消火栓', 'stop sign': '標識',
    'bench': 'ベンチ', 'bird': '鳥', 'cat': '猫', 'dog': '犬', 'horse': '馬',
    'sheep': '羊', 'cow': '牛', 'elephant': '象', 'bear': 'クマ', 'zebra': 'シマウマ',
    'giraffe': 'キリン', 'backpack': 'リュック', 'umbrella': '傘', 'handbag': 'バッグ',
    'tie': 'ネクタイ', 'suitcase': 'スーツケース', 'frisbee': 'フリスビー', 'skis': 'スキー',
    'snowboard': 'スノーボード', 'sports ball': 'ボール', 'kite': '凧', 'baseball bat': 'バット',
    'baseball glove': 'グローブ', 'skateboard': 'スケートボード', 'surfboard': 'サーフボード',
    'tennis racket': 'ラケット', 'bottle': 'ボトル', 'wine glass': 'グラス', 'cup': 'カップ',
    'fork': 'フォーク', 'knife': 'ナイフ', 'spoon': 'スプーン', 'bowl': '料理',
    'banana': 'バナナ', 'apple': 'リンゴ', 'sandwich': 'サンドイッチ', 'orange': 'オレンジ',
    'broccoli': 'ブロッコリー', 'carrot': 'ニンジン', 'hot dog': 'ホットドッグ', 'pizza': 'ピザ',
    'donut': 'ドーナツ', 'cake': 'ケーキ', 'chair': '椅子', 'couch': 'ソファ',
    'potted plant': '植物', 'bed': 'ベッド', 'dining table': 'テーブル', 'toilet': 'トイレ',
    'tv': 'テレビ', 'laptop': 'パソコン', 'mouse': 'マウス', 'remote': 'リモコン',
    'keyboard': 'キーボード', 'cell phone': 'スマホ', 'microwave': '電子レンジ',
    'oven': 'オーブン', 'toaster': 'トースター', 'sink': 'シンク', 'refrigerator': '冷蔵庫',
    'book': '本', 'clock': '時計', 'vase': '花瓶', 'scissors': 'ハサミ',
    'teddy bear': 'ぬいぐるみ', 'hair drier': 'ドライヤー', 'toothbrush': '歯ブラシ'
}

class OpenVINOYOLOEngine:
    def __init__(self, device="GPU"):
        from ultralytics import YOLO
        import openvino as ov

        self.device = device
        core = ov.Core()
        available = core.available_devices
        print(f"[OpenVINO] 検出されたデバイス: {available}", flush=True)

        if device.upper() == "GPU" and not any("GPU" in d for d in available):
            print("[OpenVINO] GPUが検出されなかったため、CPUを使用します。", flush=True)
            self.device = "CPU"
        else:
            self.device = device.upper()
            print(f"[OpenVINO] 選択されたデバイス: {self.device}", flush=True)

        model_dir = os.path.join(os.path.expanduser("~"), ".openvino_photo_rename_model")
        os.makedirs(model_dir, exist_ok=True)
        ov_model_path = os.path.join(model_dir, "yolo11n_openvino_model")

        if not os.path.exists(ov_model_path):
            print("[OpenVINO] YOLOv11n を OpenVINO IR 形式に最適化エクスポート中（初回のみ）...", flush=True)
            pt = YOLO("yolo11n.pt")
            ov_model_path = pt.export(format="openvino", dynamic=False)

        print(f"[OpenVINO] モデル読み込み中: {ov_model_path} (intel:{self.device})", flush=True)
        self.model = YOLO(ov_model_path, task="detect")
        self.intel_device_str = f"intel:{self.device}"
        
        # 初回ウォームアップ推論
        dummy_img = Image.new("RGB", (640, 640), color="white")
        try:
            self.model(dummy_img, device=self.intel_device_str, verbose=False)
        except Exception:
            self.model(dummy_img, verbose=False)

        print(f"[OpenVINO] モデル準備完了！デバイス '{self.device}' で超高速推論（0.01〜0.05秒/枚）が利用可能です。", flush=True)

    def predict_base64(self, b64_str):
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        try:
            results = self.model(img, device=self.intel_device_str, verbose=False, conf=0.35)
        except Exception:
            results = self.model(img, verbose=False, conf=0.35)

        if not results or len(results) == 0:
            return "写真"
        boxes = results[0].boxes
        if len(boxes) == 0:
            return "写真"

        detected_names = []
        person_count = 0
        food_count = 0

        for box in boxes:
            cls_id = int(box.cls[0])
            name = self.model.names[cls_id]
            detected_names.append(name)
            if name == 'person':
                person_count += 1
            elif name in ['sandwich', 'cake', 'pizza', 'donut', 'hot dog', 'bowl', 'dining table']:
                food_count += 1

        if person_count >= 3:
            return "集合写真"
        elif food_count >= 1:
            return "料理"

        counts = Counter(detected_names)
        top_cls, _ = counts.most_common(1)[0]
        return COCO_JA_MAP.get(top_cls, "写真")

class OpenVINORequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ['/health', '/api/tags', '/v1/models', '/']:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            data = {
                "status": "ok",
                "engine": "OpenVINO",
                "device": ov_engine.device if ov_engine else "unknown",
                "models": [{"name": "openvino-yolo11", "model": "openvino-yolo11"}]
            }
            self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            req = json.loads(body.decode('utf-8'))
        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON"}')
            return

        b64 = None
        if "images" in req and req["images"]:
            b64 = req["images"][0]
        elif "image" in req:
            b64 = req["image"]
        elif "messages" in req:
            for m in req["messages"]:
                if isinstance(m.get("content"), list):
                    for c in m["content"]:
                        if c.get("type") == "image_url":
                            url = c.get("image_url", {}).get("url", "")
                            if "," in url:
                                b64 = url.split(",")[1]

        if not b64:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "No image provided"}')
            return

        try:
            t0 = time.time()
            label = ov_engine.predict_base64(b64)
            elapsed = round((time.time() - t0) * 1000, 1)

            print(f"[{self.client_address[0]}] 判定: 【{label}】 ({elapsed} ms)", flush=True)

            resp_data = {
                "response": label,
                "choices": [{"message": {"content": label}}],
                "model": "openvino-yolo11"
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(resp_data, ensure_ascii=False).encode('utf-8'))
        except Exception as e:
            traceback.print_exc()
            self.send_response(500)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            err_data = {"error": str(e), "response": "写真"}
            self.wfile.write(json.dumps(err_data).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def log_message(self, format, *args):
        pass

def main():
    parser = argparse.ArgumentParser(description="OpenVINO Dedicated LAN AI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000)")
    parser.add_argument("--device", default="GPU", help="OpenVINO Device: GPU, NPU, CPU (default: GPU)")
    args = parser.parse_args()

    global ov_engine
    ov_engine = OpenVINOYOLOEngine(device=args.device)

    server = HTTPServer((args.host, args.port), OpenVINORequestHandler)
    print(f"\n=======================================================")
    print(f"=== OpenVINO Dedicated LAN AI Server ===")
    print(f"待機アドレス : http://{args.host}:{args.port}")
    print(f"稼働デバイス : {ov_engine.device} (Intel Arc GPU / NPU / CPU)")
    print(f"※ このウィンドウを開いたままにしておいてください")
    print(f"=======================================================\n", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nサーバーを停止しました。")

if __name__ == '__main__':
    main()
