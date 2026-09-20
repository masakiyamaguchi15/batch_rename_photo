import os
import sys
import re
import json
import time
import io
import argparse
import base64
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import requests
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS, IFD
import pillow_heif
try:
    import reverse_geocoder as rg
except Exception:
    rg = None
from geopy.geocoders import Nominatim

sys.stdout.reconfigure(encoding='utf-8')
pillow_heif.register_heif_opener()

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.heic', '.png'}
BASE_PHOTO_DIR = r"G:\マイドライブ\5-PHOTO"

geolocator = Nominatim(user_agent="photo_sorter_openvino_vlm_tool")
geo_cache = {}

# グローバルモデル保持
vlm_pipeline = None
yolo_model = None

# .env ファイルが存在すれば自動ロード
def _load_dotenv():
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_dotenv()

# Cloudflare Workers AI 設定 (環境変数 / .env から取得)
DEFAULT_CF_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
DEFAULT_CF_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
DEFAULT_CF_MODEL = "@cf/meta/llama-3.2-11b-vision-instruct"

COCO_JA_MAP = {
    'person': '人物', 'bicycle': '自転車', 'car': '自動車', 'motorcycle': 'バイク',
    'airplane': '飛行機', 'bus': 'バス', 'train': '電車', 'truck': 'トラック',
    'boat': '船', 'traffic light': '信号', 'fire hydrant': '消火栓', 'stop sign': '標識',
    'bench': 'ベンチ', 'bird': '鳥', 'cat': '猫', 'dog': '犬', 'horse': '馬',
    'sheep': '羊', 'cow': '牛', 'elephant': '象', 'bear': 'クマ', 'zebra': 'シマウマ',
    'giraffe': 'キリン', 'backpack': 'リュック', 'umbrella': '傘', 'handbag': 'バッグ',
    'tie': 'ネクタイ', 'suitcase': 'スーツケース', 'frisbee': 'フリスビー', 'skis': 'スキー',
    'snowboard': 'スノーボード', 'sports ball': 'ボール', 'kite': '凧', 'baseball bat': 'バット',
    'baseball glove': 'グローブ', 'skateboard': 'スケボー', 'surfboard': 'サーフボード',
    'tennis racket': 'ラケット', 'bottle': 'ボトル', 'wine glass': 'グラス', 'cup': 'コップ',
    'fork': 'フォーク', 'knife': 'ナイフ', 'spoon': 'スプーン', 'bowl': '器',
    'banana': 'バナナ', 'apple': 'リンゴ', 'sandwich': 'サンドイッチ', 'orange': 'オレンジ',
    'broccoli': '野菜', 'carrot': 'ニンジン', 'hot dog': 'ホットドッグ', 'pizza': 'ピザ',
    'donut': 'ドーナツ', 'cake': 'ケーキ', 'chair': '椅子', 'couch': 'ソファ',
    'potted plant': '観葉植物', 'bed': 'ベッド', 'dining table': 'テーブル', 'toilet': 'トイレ',
    'tv': 'テレビ', 'laptop': 'パソコン', 'mouse': 'マウス', 'remote': 'リモコン',
    'keyboard': 'キーボード', 'cell phone': 'スマホ', 'microwave': '電子レンジ',
    'oven': 'オーブン', 'toaster': 'トースター', 'sink': 'シンク', 'refrigerator': '冷蔵庫',
    'book': '本', 'clock': '時計', 'vase': '花瓶', 'scissors': 'ハサミ',
    'teddy bear': 'ぬいぐるみ', 'hair drier': 'ドライヤー', 'toothbrush': '歯ブラシ'
}

def sanitize_filename_part(text):
    if not text:
        return "写真"
    first_line = text.strip().split('\n')[0].split('。')[0]
    # 「この写真の主要な被写体は」「主要な被写体:」「答え:」などの前置き・オウム返しを除去
    first_line = re.sub(r'^(この写真の主要な被写体|この写真の被写体|この写真|主要な被写体|主要な情景|被写体|情景|写真の内容|回答|答え|画像の内容|判定)[:：\sはが]+', '', first_line)
    first_line = re.sub(r'^[0-9]+[\.\)\s、]+', '', first_line)
    first_line = re.sub(r'(です|である|の写真)$', '', first_line)
    cleaned = re.sub(r'[\\/*?:"<>|\[\]\(\)\s　、。,\.\-\_]+', '', first_line)
    cleaned = cleaned[:12]
    return cleaned if cleaned else "写真"

def check_ollama_status(ollama_url="http://localhost:11434"):
    """Ollama サーバーの導通状態を確認"""
    try:
        target_url = ollama_url.replace("0.0.0.0", "127.0.0.1").rstrip('/')
        resp = requests.get(f"{target_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False

def recognize_content_vlm(file_path, model="llama3.2-vision", ollama_url="http://localhost:11434", timeout=180):
    """Ollama Vision API (llama3.2-vision / llava など) で日本語表現を取得"""
    try:
        target_url = ollama_url.replace("0.0.0.0", "127.0.0.1").rstrip('/')
        with Image.open(file_path) as im:
            im_rgb = im.convert("RGB")
            im_rgb.thumbnail((1024, 1024))
            buf = io.BytesIO()
            im_rgb.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()
            b64_image = base64.b64encode(img_bytes).decode("utf-8")

        prompt = (
            "この写真の主要な被写体または情景を、ファイル名に使用できる簡潔な日本語の名詞"
            "（1〜2単語程度、記号なし、2〜8文字、例: ガジュマル, ジンベエザメ, ハイビスカス, 初日の出, ダイビング, BBQ, 集合写真, 料理 など）"
            "で1つだけ出力してください。説明文や英語は不要です。"
        )

        payload = {
            "model": model,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 30
            }
        }

        for attempt in range(2):
            try:
                resp = requests.post(f"{target_url}/api/generate", json=payload, timeout=timeout)
                if resp.status_code == 200:
                    res_data = resp.json()
                    raw_text = res_data.get("response", "").strip()
                    raw_text = raw_text.split('\n')[0].split('。')[0]
                    return sanitize_filename_part(raw_text)
                else:
                    print(f"Ollama API エラー (HTTP {resp.status_code}): {resp.text}", flush=True)
            except requests.exceptions.Timeout:
                if attempt == 0:
                    print(f"    [再試行] Ollama 応答タイムアウト（{timeout}秒）。モデル準備中の可能性があるため再試行します...", flush=True)
                    time.sleep(2)
                    continue
                else:
                    print(f"    [タイムアウト] Ollama 応答が制限時間内に返りませんでした。", flush=True)
            except Exception as e:
                print(f"VLM 通信エラー ({os.path.basename(file_path)}): {e}", flush=True)
                break

        return recognize_content_yolo(file_path)

    except Exception as e:
        print(f"VLM 推論エラー ({os.path.basename(file_path)}): {e}", flush=True)
        return recognize_content_yolo(file_path)

def check_lmstudio_status(lmstudio_url="http://localhost:1234"):
    """LM Studio (OpenAI互換) サーバーの導通状態を確認"""
    try:
        target_url = lmstudio_url.replace("0.0.0.0", "127.0.0.1").rstrip('/')
        resp = requests.get(f"{target_url}/v1/models", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False

def recognize_content_lmstudio(file_path, model="default", lmstudio_url="http://localhost:1234", timeout=180):
    """LM Studio (OpenAI 互換 Vision API) で日本語表現を取得"""
    try:
        target_url = lmstudio_url.replace("0.0.0.0", "127.0.0.1").rstrip('/')
        with Image.open(file_path) as im:
            im_rgb = im.convert("RGB")
            im_rgb.thumbnail((1024, 1024))
            buf = io.BytesIO()
            im_rgb.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()
            b64_image = base64.b64encode(img_bytes).decode("utf-8")

        prompt = (
            "この写真の主要な被写体または情景を、ファイル名に使用できる簡潔な日本語の名詞"
            "（1〜2単語程度、記号なし、2〜8文字、例: ガジュマル, ジンベエザメ, ハイビスカス, 初日の出, ダイビング, BBQ, 集合写真, 料理 など）"
            "で1つだけ出力してください。説明文や英語は不要です。"
        )

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 30
        }

        for attempt in range(2):
            try:
                resp = requests.post(f"{target_url}/v1/chat/completions", json=payload, timeout=timeout)
                if resp.status_code == 200:
                    res_data = resp.json()
                    raw_text = res_data["choices"][0]["message"]["content"].strip()
                    raw_text = raw_text.split('\n')[0].split('。')[0]
                    return sanitize_filename_part(raw_text)
                else:
                    print(f"LM Studio API エラー (HTTP {resp.status_code}): {resp.text}", flush=True)
            except requests.exceptions.Timeout:
                if attempt == 0:
                    print(f"    [再試行] LM Studio 応答タイムアウト（{timeout}秒）。再試行中...", flush=True)
                    time.sleep(2)
                    continue
                else:
                    print(f"    [タイムアウト] LM Studio 応答が制限時間内に返りませんでした。", flush=True)
            except Exception as e:
                print(f"LM Studio 通信エラー ({os.path.basename(file_path)}): {e}", flush=True)
                break

        return recognize_content_yolo(file_path)
    except Exception as e:
        print(f"LM Studio 画像処理エラー ({os.path.basename(file_path)}): {e}", flush=True)
        return recognize_content_yolo(file_path)

def check_cloudflare_status(account_id, api_token, model="@cf/meta/llama-3.2-11b-vision-instruct"):
    """Cloudflare Workers AI の導通状態を確認し、利用規約同意（agree）を行う"""
    try:
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        resp = requests.post(url, headers=headers, json={"prompt": "agree"}, timeout=15)
        if resp.status_code == 200:
            return True
        if resp.status_code == 403 and "agree" in resp.text.lower():
            return True
        return False
    except Exception:
        return False

def recognize_content_cloudflare(file_path, account_id, api_token, model="@cf/meta/llama-3.2-11b-vision-instruct", timeout=60):
    """Cloudflare Workers AI (Llama 3.2 11B Vision) で日本語表現を取得"""
    try:
        with Image.open(file_path) as im:
            im_rgb = im.convert("RGB")
            im_rgb.thumbnail((1024, 1024))
            buf = io.BytesIO()
            im_rgb.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()
            b64_image = base64.b64encode(img_bytes).decode("utf-8")

        prompt = (
            "この写真の主要な被写体または情景を、ファイル名に使用できる簡潔な日本語の名詞"
            "（1〜2単語程度、記号なし、2〜8文字、例: ガジュマル, ジンベエザメ, ハイビスカス, 初日の出, ダイビング, BBQ, 集合写真, 料理 など）"
            "で1つだけ出力してください。説明文や英語は不要です。"
        )

        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
            "image": b64_image,
            "max_tokens": 30
        }

        for attempt in range(3):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    res_data = resp.json()
                    raw_text = res_data.get("result", {}).get("response", "").strip()
                    raw_text = raw_text.split('\n')[0].split('。')[0]
                    time.sleep(0.3)  # レートリミット防止の微小ウェイト
                    return sanitize_filename_part(raw_text)
                elif resp.status_code == 429:
                    wait_sec = 2 * (attempt + 1)
                    print(f"    [Cloudflare レート制限] 429 Too Many Requests。{wait_sec}秒待機後に再試行...", flush=True)
                    time.sleep(wait_sec)
                    continue
                else:
                    print(f"Cloudflare Workers AI エラー (HTTP {resp.status_code}): {resp.text}", flush=True)
            except requests.exceptions.Timeout:
                if attempt < 2:
                    print(f"    [再試行] Cloudflare 応答タイムアウト（{timeout}秒）。再試行中...", flush=True)
                    time.sleep(2)
                    continue
                else:
                    print(f"    [タイムアウト] Cloudflare 応答が制限時間内に返りませんでした。", flush=True)
            except Exception as e:
                print(f"Cloudflare 通信エラー ({os.path.basename(file_path)}): {e}", flush=True)
                break

        return recognize_content_yolo(file_path)
    except Exception as e:
        print(f"Cloudflare 画像処理エラー ({os.path.basename(file_path)}): {e}", flush=True)
        return recognize_content_yolo(file_path)

def load_yolo_model():
    global yolo_model
    if yolo_model is not None:
        return yolo_model

    try:
        from ultralytics import YOLO
    except ImportError:
        return None

    try:
        model_dir = os.path.join(os.path.expanduser("~"), ".openvino_photo_rename_model")
        os.makedirs(model_dir, exist_ok=True)
        ov_export_path = os.path.join(model_dir, "yolo11n_openvino_model")

        if not os.path.exists(ov_export_path):
            pt_model = YOLO("yolo11n.pt")
            ov_export_path = pt_model.export(format="openvino")

        yolo_model = YOLO(ov_export_path, task="detect")
        return yolo_model
    except Exception as e:
        print(f"YOLO モデル読み込みスキップ: {e}", flush=True)
        return None

def recognize_content_yolo(file_path):
    try:
        model = load_yolo_model()
        if model is None:
            return "写真"
        results = model(file_path, verbose=False, conf=0.35)
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
            name = model.names[cls_id]
            detected_names.append(name)
            if name == 'person':
                person_count += 1
            elif name in ['sandwich', 'cake', 'pizza', 'donut', 'hot dog', 'bowl']:
                food_count += 1

        if person_count >= 3:
            return "集合写真"
        elif food_count >= 1:
            return "料理"

        counts = Counter(detected_names)
        most_common_eng, _ = counts.most_common(1)[0]
        ja_name = COCO_JA_MAP.get(most_common_eng, "写真")
        return sanitize_filename_part(ja_name)
    except Exception:
        return "写真"

PREF_EN_JA = {
    'Hokkaido': '北海道', 'Aomori': '青森県', 'Iwate': '岩手県', 'Miyagi': '宮城県', 'Akita': '秋田県',
    'Yamagata': '山形県', 'Fukushima': '福島県', 'Ibaraki': '茨城県', 'Tochigi': '栃木県', 'Gunma': '群馬県',
    'Saitama': '埼玉県', 'Chiba': '千葉県', 'Tokyo': '東京都', 'Kanagawa': '神奈川県', 'Niigata': '新潟県',
    'Toyama': '富山県', 'Ishikawa': '石川県', 'Fukui': '福井県', 'Yamanashi': '山梨県', 'Nagano': '長野県',
    'Gifu': '岐阜県', 'Shizuoka': '静岡県', 'Aichi': '愛知県', 'Mie': '三重県', 'Shiga': '滋賀県',
    'Kyoto': '京都府', 'Osaka': '大阪府', 'Hyogo': '兵庫県', 'Nara': '奈良県', 'Wakayama': '和歌山県',
    'Tottori': '鳥取県', 'Shimane': '島根県', 'Okayama': '岡山県', 'Hiroshima': '広島県', 'Yamaguchi': '山口県',
    'Tokushima': '徳島県', 'Kagawa': '香川県', 'Ehime': '愛媛県', 'Kochi': '高知県', 'Fukuoka': '福岡県',
    'Saga': '佐賀県', 'Nagasaki': '長崎県', 'Kumamoto': '熊本県', 'Oita': '大分県', 'Miyazaki': '宮崎県',
    'Kagoshima': '鹿児島県', 'Okinawa': '沖縄県'
}

def get_location_name(lat, lon):
    if lat is None or lon is None:
        return ""
    cache_key = f"{round(lat, 3)},{round(lon, 3)}"
    if cache_key in geo_cache:
        return geo_cache[cache_key]
    
    # 1. 超高速なオフライン逆ジオコーディング (0.1ミリ秒 / ネットワーク待機ゼロ)
    if rg:
        try:
            res = rg.search([(lat, lon)], mode=1)[0]
            pref = res.get('admin1', '').strip()
            name = res.get('name', '').strip()
            pref_ja = PREF_EN_JA.get(pref, pref)
            loc_str = f"{pref_ja}{name}".strip()
            if loc_str:
                geo_cache[cache_key] = loc_str
                return loc_str
        except Exception:
            pass

    return ""

def convert_dms_to_deg(dms, ref):
    if not dms or len(dms) < 3:
        return None
    try:
        d = float(dms[0])
        m = float(dms[1])
        s = float(dms[2])
        deg = d + (m / 60.0) + (s / 3600.0)
        if ref in ['S', 'W']:
            deg = -deg
        return deg
    except Exception:
        return None

def extract_exif_info(file_path):
    date_str = None
    time_sort = 0
    lat = None
    lon = None
    
    try:
        with Image.open(file_path) as img:
            exif = img.getexif()
            if exif:
                dt_raw = None
                try:
                    exif_ifd = exif.get_ifd(IFD.Exif)
                    dt_raw = exif_ifd.get(36867) or exif_ifd.get(36868)
                except Exception:
                    pass
                if not dt_raw:
                    dt_raw = exif.get(306)
                
                if dt_raw and isinstance(dt_raw, str):
                    m = re.search(r'(\d{4})[:\-](\d{2})[:\-](\d{2})\s+(\d{2})[:\-](\d{2})[:\-](\d{2})', dt_raw)
                    if m:
                        date_str = f"{m.group(1)}{m.group(2)}{m.group(3)}"
                        time_sort = int(f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4)}{m.group(5)}{m.group(6)}")
                    else:
                        m2 = re.search(r'(\d{4})[:\-](\d{2})[:\-](\d{2})', dt_raw)
                        if m2:
                            date_str = f"{m2.group(1)}{m2.group(2)}{m2.group(3)}"
                
                try:
                    gps_ifd = exif.get_ifd(IFD.GPSInfo)
                    if gps_ifd:
                        lat_val = gps_ifd.get(2)
                        lat_ref = gps_ifd.get(1, 'N')
                        lon_val = gps_ifd.get(4)
                        lon_ref = gps_ifd.get(3, 'E')
                        if lat_val and lon_val:
                            lat = convert_dms_to_deg(lat_val, lat_ref)
                            lon = convert_dms_to_deg(lon_val, lon_ref)
                except Exception:
                    pass
    except Exception:
        pass
    
    if not date_str:
        parent = os.path.basename(os.path.dirname(file_path))
        m_parent = re.search(r'(\d{8})', parent)
        if m_parent:
            date_str = m_parent.group(1)
        else:
            fname = os.path.basename(file_path)
            m_fname = re.search(r'(20\d{6})', fname)
            if m_fname:
                date_str = m_fname.group(1)
            else:
                m_fname2 = re.search(r'(\d{2})(\d{2})(\d{2})', fname)
                if m_fname2 and 0 <= int(m_fname2.group(1)) <= 30:
                    date_str = f"20{m_fname2.group(1)}{m_fname2.group(2)}{m_fname2.group(3)}"
        
        if not date_str:
            try:
                mtime = os.path.getmtime(file_path)
                dt = datetime.fromtimestamp(mtime)
                date_str = dt.strftime("%Y%m%d")
                time_sort = int(dt.strftime("%Y%m%d%H%M%S"))
            except Exception:
                date_str = "20090101"
                
    if time_sort == 0:
        try:
            mtime = os.path.getmtime(file_path)
            time_sort = int(datetime.fromtimestamp(mtime).strftime("%Y%m%d%H%M%S"))
        except Exception:
            time_sort = 0
            
    return date_str, time_sort, lat, lon

def scan_files(root_dir):
    image_files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VALID_EXTENSIONS:
                if f.startswith("batch_rename") or f.startswith("rollback"):
                    continue
                image_files.append(os.path.join(root, f))
    return image_files

def select_year_interactively():
    if not os.path.exists(BASE_PHOTO_DIR):
        print(f"[ERROR] 写真ルートフォルダーが見つかりません: {BASE_PHOTO_DIR}")
        sys.exit(1)

    entries = os.listdir(BASE_PHOTO_DIR)
    year_dirs = [e for e in entries if os.path.isdir(os.path.join(BASE_PHOTO_DIR, e)) and re.match(r'^\d{4}$', e)]
    year_dirs.sort()

    if not year_dirs:
        print(f"[ERROR] 年別フォルダーが見つかりませんでした: {BASE_PHOTO_DIR}")
        sys.exit(1)

    print("\n=== 対象の年別フォルダーを選択してください ===")
    for idx, y in enumerate(year_dirs, start=1):
        full_p = os.path.join(BASE_PHOTO_DIR, y)
        file_cnt = len(scan_files(full_p))
        print(f" [{idx}] {y}年 ({file_cnt} 枚の画像)")

    print(" ===========================================")
    while True:
        try:
            choice = input(f"番号を入力してください (1-{len(year_dirs)}): ").strip()
            if choice.isdigit():
                num = int(choice)
                if 1 <= num <= len(year_dirs):
                    selected_year = year_dirs[num - 1]
                    return os.path.join(BASE_PHOTO_DIR, selected_year)
        except (KeyboardInterrupt, EOFError):
            print("\n処理をキャンセルしました。")
            sys.exit(0)
        print("有効な番号を入力してください。")

def main():
    default_ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if "0.0.0.0" in default_ollama_host:
        default_ollama_host = default_ollama_host.replace("0.0.0.0", "127.0.0.1")
    if not default_ollama_host.startswith("http://") and not default_ollama_host.startswith("https://"):
        default_ollama_host = f"http://{default_ollama_host}"
    if ":11434" not in default_ollama_host and not default_ollama_host.endswith(":11434"):
        default_ollama_host = f"{default_ollama_host}:11434"

    parser = argparse.ArgumentParser(description="Rename photos with EXIF date, location, and Vision AI Model (LM Studio / Ollama).")
    parser.add_argument("--year", help="Target year (e.g. 2009, 2025)")
    parser.add_argument("--root", help="Root directory (Overrides --year)")
    parser.add_argument("--execute", action="store_true", help="Perform actual rename")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files for testing")
    parser.add_argument("--model", default="default", help="AI Model to use (default: auto-detected or llava:13b)")
    parser.add_argument("--openvino-url", help="OpenVINO Server URL (e.g. http://192.168.40.116:8000)")
    parser.add_argument("--lmstudio-url", help="LM Studio Server URL (e.g. http://192.168.40.116:1234)")
    parser.add_argument("--ollama-url", default=default_ollama_host, help=f"Ollama Server URL (default: {default_ollama_host})")
    parser.add_argument("--cloudflare", action="store_true", help="Use Cloudflare Workers AI")
    parser.add_argument("--cf-account-id", default=DEFAULT_CF_ACCOUNT_ID, help="Cloudflare Account ID (from .env or CLOUDFLARE_ACCOUNT_ID)")
    parser.add_argument("--cf-token", default=DEFAULT_CF_API_TOKEN, help="Cloudflare API Token (from .env or CLOUDFLARE_API_TOKEN)")
    parser.add_argument("--cf-model", default=DEFAULT_CF_MODEL, help=f"Cloudflare AI Vision Model (default: {DEFAULT_CF_MODEL})")
    parser.add_argument("--timeout", type=int, default=180, help="API timeout in seconds (default: 180)")
    args = parser.parse_args()

    if args.root:
        target_root = args.root
    elif args.year:
        target_root = os.path.join(BASE_PHOTO_DIR, str(args.year))
    else:
        target_root = select_year_interactively()

    if not os.path.exists(target_root):
        print(f"[ERROR] ディレクトリが存在しません: {target_root}")
        sys.exit(1)

    history_file = os.path.join(target_root, "rename_history.json")
    csv_file = os.path.join(target_root, "rename_preview.csv")
    cache_file = os.path.join(target_root, "photo_cache.json")

    # バックエンド判定 (Cloudflare または OpenVINO または LM Studio または Ollama または YOLO)
    backend = "ollama"
    server_url = ""
    is_server_active = False

    if args.cloudflare or (args.model and "@cf/" in args.model):
        backend = "cloudflare"
        server_url = f"https://api.cloudflare.com/.../ai/run/{args.cf_model}"
        print(f"[Cloudflare] Workers AI 接続を確認中 (モデル: {args.cf_model})...", flush=True)
        is_server_active = check_cloudflare_status(args.cf_account_id, args.cf_token, args.cf_model)
        if is_server_active:
            model_label = f"Cloudflare ({os.path.basename(args.cf_model)})"
            print(f"[Cloudflare] 接続確認完了！高精度クラウドGPU推論が可能です。\n", flush=True)
        else:
            print(f"[WARNING] Cloudflare Workers AI への接続に失敗しました。")
            print("          トークンやAccount IDを確認してください。ローカル YOLO モデルへフォールバックします。")
            model_label = "Cloudflare (未接続)"
    elif args.openvino_url or (args.ollama_url and ":8000" in args.ollama_url):
        backend = "openvino"
        server_url = args.openvino_url if args.openvino_url else args.ollama_url
        if not server_url.startswith("http"):
            server_url = f"http://{server_url}"
        try:
            chk_url = server_url.replace("0.0.0.0", "127.0.0.1").rstrip('/')
            resp = requests.get(f"{chk_url}/health", timeout=5)
            is_server_active = (resp.status_code == 200)
            if is_server_active:
                info = resp.json()
                engine_name = info.get('model_name', info.get('device', 'GPU'))
                model_label = f"OpenVINO ({engine_name})"
            else:
                model_label = "OpenVINO (GPU)"
        except Exception:
            is_server_active = False
            model_label = "OpenVINO (GPU)"

        if not is_server_active:
            print(f"[WARNING] OpenVINO サーバー ({server_url}) に接続できませんでした。")
            print("          サーバーが未起動の場合はローカル YOLO モデルへフォールバックします。")
    elif args.lmstudio_url or (args.ollama_url and ":1234" in args.ollama_url):
        backend = "lmstudio"
        server_url = args.lmstudio_url if args.lmstudio_url else args.ollama_url
        if not server_url.startswith("http"):
            server_url = f"http://{server_url}"
        is_server_active = check_lmstudio_status(server_url)
        if not is_server_active:
            print(f"[WARNING] LM Studio サーバー ({server_url}) に接続できませんでした。")
            print("          LM Studio が未起動の場合は YOLO モデルへフォールバックします。")
        else:
            try:
                target_chk = server_url.replace("0.0.0.0", "127.0.0.1").rstrip('/')
                m_resp = requests.get(f"{target_chk}/v1/models", timeout=5).json()
                loaded_models = [m.get("id") for m in m_resp.get("data", [])]
                if loaded_models:
                    if args.model in ["default", "llava:13b", "llava:34b"]:
                        args.model = loaded_models[0]
            except Exception:
                pass
        model_label = f"LM Studio ({args.model})"
    elif args.model.lower() == "yolo":
        backend = "yolo"
        is_server_active = True
        model_label = "YOLOv11"
    else:
        backend = "ollama"
        if args.model == "default":
            args.model = "llava:13b"
        server_url = args.ollama_url
        target_check_url = server_url.replace("0.0.0.0", "127.0.0.1")
        is_server_active = check_ollama_status(target_check_url)
        if not is_server_active:
            print(f"[WARNING] Ollama サーバー ({server_url}) に接続できませんでした。")
            print("          Ollama が未起動の場合は YOLO モデルへフォールバックします。")
        model_label = f"Ollama ({args.model})"

    print(f"\n==========================================")
    print(f"=== Photo Batch Renaming Tool ===")
    print(f"AI バックエンド  : {backend.upper()}")
    print(f"使用AIモデル     : {model_label}")
    if backend != "yolo":
        print(f"AI サーバー URL  : {server_url}")
    print(f"対象ディレクトリ : {target_root}")
    print(f"動作モード       : {'【実行 (ファイル変更あり)】' if args.execute else '【プレビューのみ (DRY-RUN)】'}")
    if args.limit > 0:
        print(f"処理件数制限     : 上限 {args.limit} 件")
    print(f"==========================================\n")

    if backend == "ollama" and is_server_active:
        print(f"[Ollama] モデル '{args.model}' を初期化中（大容量モデルの初回起動は数十秒待機します）...", flush=True)
        try:
            target_warmup_url = server_url.replace("0.0.0.0", "127.0.0.1").rstrip('/')
            requests.post(
                f"{target_warmup_url}/api/generate",
                json={"model": args.model, "prompt": "", "keep_alive": "15m"},
                timeout=args.timeout
            )
            print(f"[Ollama] モデル '{args.model}' の準備が完了しました。\n", flush=True)
        except Exception as e:
            print(f"[Ollama] 事前確認完了 (継続します)\n", flush=True)
    elif backend == "lmstudio" and is_server_active:
        print(f"[LM Studio] 接続確認完了。アクティブモデル: {args.model}\n", flush=True)

    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"キャッシュデータ: {len(cache)} 件のメタデータをロードしました。")
        except Exception:
            pass

    files = scan_files(target_root)
    print(f"対象画像ファイル数: 全 {len(files)} 件")
    if len(files) == 0:
        print("対象となる画像ファイルが見つかりませんでした。終了します。")
        return

    if args.limit > 0:
        files = files[:args.limit]

    dir_files = defaultdict(list)
    for f in files:
        dir_files[os.path.dirname(f)].append(f)

    all_rename_plans = []
    total_count = len(files)
    processed_count = 0

    folder_data = {}
    print(f"[1/2] EXIF・撮影日時・GPS位置情報の解析を開始します（全 {total_count} 件）...", flush=True)
    t_exif_start = time.time()
    exif_done = 0
    cached_exif_count = 0

    for d, flist in dir_files.items():
        dir_records = []
        gps_locations_in_dir = []
        
        # 1st Pass: EXIF / GPS (キャッシュがあればクラウド・ディスクアクセスをスキップ)
        for fpath in flist:
            exif_done += 1
            if exif_done % 500 == 0 or exif_done == total_count:
                print(f"  -> EXIF解析進捗: [{exif_done}/{total_count}]", flush=True)

            rel = os.path.relpath(fpath, target_root)
            cached_entry = cache.get(fpath) or cache.get(os.path.normpath(fpath))

            if cached_entry and cached_entry.get("date"):
                # キャッシュから高速ロード (Google ドライブ等の通信待機ゼロ)
                date_str = cached_entry.get("date")
                loc_str = cached_entry.get("location", "")
                time_sort = cached_entry.get("time_sort", 0)
                lat = cached_entry.get("lat")
                lon = cached_entry.get("lon")
                if not time_sort and date_str:
                    try:
                        time_sort = int(date_str + "000000")
                    except Exception:
                        time_sort = 0
                cached_exif_count += 1
            else:
                date_str, time_sort, lat, lon = extract_exif_info(fpath)
                loc_str = ""
                if lat is not None and lon is not None:
                    loc_str = get_location_name(lat, lon)

            if loc_str:
                gps_locations_in_dir.append(loc_str)

            dir_records.append({
                "path": fpath,
                "rel": rel,
                "date": date_str,
                "time_sort": time_sort,
                "lat": lat,
                "lon": lon,
                "location": loc_str,
                "ext": os.path.splitext(fpath)[1]
            })

        default_loc = ""
        if gps_locations_in_dir:
            default_loc = Counter(gps_locations_in_dir).most_common(1)[0][0]
        folder_data[d] = (dir_records, default_loc)

    t_exif_elapsed = round(time.time() - t_exif_start, 1)
    print(f"  -> EXIF解析完了 (所要時間: {t_exif_elapsed} 秒, キャッシュ即時ロード: {cached_exif_count}/{total_count} 件)\n", flush=True)
    print(f"[2/2] AI 画像認識およびリネーム名判定を開始します...", flush=True)

    for d, (dir_records, default_loc) in folder_data.items():
        # 2nd Pass: AI 画像認識
        for item in dir_records:
            processed_count += 1
            fpath = item["path"]
            
            if not item["location"]:
                item["location"] = default_loc

            cached_entry = cache.get(fpath)
            if cached_entry and "content" in cached_entry and cached_entry["content"]:
                content = cached_entry["content"]
            else:
                print(f"[{processed_count}/{total_count}] [{model_label}] AI解析中: {os.path.basename(fpath)} ...", flush=True)
                if backend == "cloudflare" and is_server_active:
                    content = recognize_content_cloudflare(fpath, account_id=args.cf_account_id, api_token=args.cf_token, model=args.cf_model, timeout=args.timeout)
                elif backend == "lmstudio" and is_server_active:
                    content = recognize_content_lmstudio(fpath, model=args.model, lmstudio_url=server_url, timeout=args.timeout)
                elif backend in ["ollama", "openvino"] and is_server_active:
                    content = recognize_content_vlm(fpath, model=args.model, ollama_url=server_url, timeout=args.timeout)
                else:
                    content = recognize_content_yolo(fpath)

                print(f"    => 判定結果: 【{content}】", flush=True)
                cache[fpath] = {
                    "date": item["date"],
                    "location": item["location"],
                    "content": content,
                    "time_sort": item.get("time_sort", 0),
                    "lat": item.get("lat"),
                    "lon": item.get("lon")
                }
                with open(cache_file, "w", encoding="utf-8") as cf:
                    json.dump(cache, cf, ensure_ascii=False, indent=2)

            item["content"] = content

        with open(cache_file, "w", encoding="utf-8") as cf:
            json.dump(cache, cf, ensure_ascii=False, indent=2)

        dir_records.sort(key=lambda x: (x["time_sort"], x["path"]))

        prefix_groups = defaultdict(list)
        for item in dir_records:
            loc_part = f"_{item['location']}" if item['location'] else ""
            prefix = f"{item['date']}{loc_part}_{item['content']}"
            prefix_groups[prefix].append(item)

        for prefix, items in prefix_groups.items():
            for idx, item in enumerate(items, start=1):
                new_filename = f"{prefix}_{idx:02d}{item['ext']}"
                new_path = os.path.join(d, new_filename)
                all_rename_plans.append({
                    "original_path": item["path"],
                    "new_path": new_path,
                    "original_name": os.path.basename(item["path"]),
                    "new_name": new_filename,
                    "dir": d
                })

    import csv
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Directory", "Original Name", "New Name", "Original Path", "New Path"])
        for p in all_rename_plans:
            writer.writerow([p["dir"], p["original_name"], p["new_name"], p["original_path"], p["new_path"]])

    print(f"\nプレビュー結果を CSV に保存しました: {csv_file}")

    rollback_script_path = os.path.join(target_root, "rollback_rename.py")
    with open(rollback_script_path, "w", encoding="utf-8") as rf:
        rf.write(f'''# 年別自動生成ロールバックスクリプト
import json
import os

HISTORY_FILE = r"{history_file}"

def rollback():
    if not os.path.exists(HISTORY_FILE):
        print(f"履歴ファイルが見つかりません: {{HISTORY_FILE}}")
        return
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    print(f"{{len(history)}} 件のファイルを元の名前へ復元中...")
    count = 0
    for item in history:
        new_path = item["new_path"]
        orig_path = item["original_path"]
        if os.path.exists(new_path):
            try:
                os.rename(new_path, orig_path)
                count += 1
            except Exception as e:
                print(f"復元エラー {{new_path}} -> {{orig_path}}: {{e}}")
    print(f"正常に {{count}} 件のファイルを元に戻しました。")

if __name__ == '__main__':
    rollback()
''')
    print(f"ロールバックスクリプトを生成しました: {rollback_script_path}")

    print("\n--- 変換プレビューサンプル（先頭 15 件） ---")
    for p in all_rename_plans[:15]:
        print(f"  {p['original_name']}  ==>  {p['new_name']}")
    if len(all_rename_plans) > 15:
        print(f"  ...他 {len(all_rename_plans) - 15} 件")

    if args.execute:
        print("\n>>> リネーム処理を開始します <<<")
        success_history = []
        renamed_count = 0
        for p in all_rename_plans:
            orig = p["original_path"]
            new_p = p["new_path"]
            if orig == new_p:
                continue
            if os.path.exists(new_p) and new_p.lower() != orig.lower():
                print(f"警告: 変更先のファイル名が既に存在するためスキップします: {new_p}")
                continue
            try:
                os.rename(orig, new_p)
                success_history.append(p)
                renamed_count += 1
            except Exception as e:
                print(f"リネーム失敗 {orig} -> {new_p}: {e}")

        with open(history_file, "w", encoding="utf-8") as hf:
            json.dump(success_history, hf, ensure_ascii=False, indent=2)
        print(f"\n==========================================")
        print(f"完了! {renamed_count} 件のファイル名を変更しました。")
        print(f"リネーム履歴: {history_file}")
        print(f"==========================================")
    else:
        print("\n※ 現在は DRY-RUN (プレビュー) モードです。ファイル名は変更されていません。")
        print(f"実際に変更を実行するには --execute オプションを付けて実行してください:")
        print(f"  python batch_rename_photos.py --year {os.path.basename(target_root)} --execute\n")

if __name__ == '__main__':
    main()
