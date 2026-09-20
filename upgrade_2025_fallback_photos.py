import os
import sys
import re
import json
import time
import io
import base64
import subprocess
from collections import defaultdict
from PIL import Image
import pillow_heif

sys.stdout.reconfigure(encoding='utf-8')
pillow_heif.register_heif_opener()

API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
MODEL = "gemini-flash-latest"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

HISTORY_FILE = r"G:\マイドライブ\5-PHOTO\2025\rename_history.json"
CACHE_FILE = r"C:\Users\yamachan\.gemini\antigravity-cli\brain\172bb532-7471-4538-b86e-399614703408\scratch\photo_cache.json"

def sanitize_filename_part(text):
    if not text:
        return "写真"
    cleaned = re.sub(r'[\\/*?:"<>|\[\]\(\)\s　、。,\.\-\_]+', '', text)
    cleaned = cleaned[:12]
    return cleaned if cleaned else "写真"

def recognize_content_gemini(file_path):
    try:
        with Image.open(file_path) as im:
            im.thumbnail((600, 600))
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=80)
            b64_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Failed to load image: {file_path}, err: {e}", flush=True)
        return "写真"

    prompt = (
        "この写真の主要な被写体または情景を、ファイル名に使用できる簡潔な日本語の名詞"
        "（1〜2単語程度、記号や空白なし、2〜8文字、例: エイ, ガジュマル, ジンベエザメ, ハイビスカス, 初日の出, ダイビング, 料理, 集合写真 など）"
        "で1つだけ出力してください。余計な説明や記号、句読点は一切出力しないでください。"
    )

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": b64_data
                    }
                }
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 100
        }
    }

    cmd = [
        "curl.exe", "-s", "-X", "POST", API_URL,
        "-H", "Content-Type: application/json",
        "--data-binary", "@-"
    ]

    try:
        res = subprocess.run(cmd, input=json.dumps(payload), capture_output=True, text=True, encoding="utf-8", timeout=20)
        if res.returncode == 0 and res.stdout:
            data = json.loads(res.stdout)
            if 'candidates' in data and len(data['candidates']) > 0:
                parts = data['candidates'][0].get('content', {}).get('parts', [])
                if parts and 'text' in parts[0]:
                    return sanitize_filename_part(parts[0]['text'].strip())
            if 'error' in data:
                print(f"API Error: {data['error'].get('message')}", flush=True)
    except Exception as e:
        print(f"Subprocess error: {e}", flush=True)

    return "写真"

def main():
    print("=== Upgrading 2025 Fallback Photos with Charged API Key ===", flush=True)
    if not os.path.exists(HISTORY_FILE):
        print("History file not found!", flush=True)
        return

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    # Find items that have '_写真_' in new_name
    fallback_items = []
    for item in history:
        if "_写真_" in item.get("new_name", ""):
            fallback_items.append(item)

    print(f"Found {len(fallback_items)} fallback items to upgrade.", flush=True)

    # Load existing cache to update it
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as cf:
                cache = json.load(cf)
        except Exception:
            pass

    # Process each file
    dir_groups = defaultdict(list)
    total = len(fallback_items)

    for idx, item in enumerate(fallback_items, start=1):
        cur_path = item["new_path"]
        if not os.path.exists(cur_path):
            continue

        print(f"[{idx}/{total}] AI Analyzing: {os.path.basename(cur_path)} ...", flush=True)
        label = recognize_content_gemini(cur_path)
        print(f"    => Label: {label}", flush=True)

        item["upgraded_label"] = label
        dir_groups[item["dir"]].append(item)

        # Update cache
        orig = item["original_path"]
        if orig in cache:
            cache[orig]["content"] = label

        time.sleep(0.5) # Fast 0.5s pause for paid tier

    # Save updated cache
    with open(CACHE_FILE, "w", encoding="utf-8") as cf:
        json.dump(cache, cf, ensure_ascii=False, indent=2)

    # Perform rename by directory
    print("\nExecuting rename of upgraded files...", flush=True)
    renamed_count = 0

    for d, items in dir_groups.items():
        # Parse existing prefix before '_写真_'
        # Format: {PREFIX}_写真_{NN}.{ext}
        prefix_map = defaultdict(list)
        for it in items:
            m = re.match(r'^(.*)_写真_\d+(\.[^.]+)$', it["new_name"])
            if m:
                base_pre = m.group(1) # e.g. 20250619_沖縄県本部町
                ext = m.group(2)
                lbl = it.get("upgraded_label", "写真")
                new_prefix = f"{base_pre}_{lbl}"
                prefix_map[(new_prefix, ext)].append(it)
            else:
                # Fallback pattern
                lbl = it.get("upgraded_label", "写真")
                ext = os.path.splitext(it["new_name"])[1]
                prefix_map[(lbl, ext)].append(it)

        # Assign numbering
        for (new_pre, ext), item_list in prefix_map.items():
            for seq, it in enumerate(item_list, start=1):
                new_final_name = f"{new_pre}_{seq:02d}{ext}"
                final_path = os.path.join(d, new_final_name)
                cur_path = it["new_path"]

                if cur_path == final_path:
                    continue

                if os.path.exists(final_path):
                    base, e = os.path.splitext(final_path)
                    final_path = f"{base}_alt{e}"
                    new_final_name = os.path.basename(final_path)

                try:
                    os.rename(cur_path, final_path)
                    it["new_path"] = final_path
                    it["new_name"] = new_final_name
                    renamed_count += 1
                except Exception as e:
                    print(f"Error renaming {cur_path} -> {final_path}: {e}", flush=True)

    # Update history file
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n==========================================")
    print(f"SUCCESS! Upgraded and renamed {renamed_count} files.")
    print(f"Updated history in: {HISTORY_FILE}")
    print(f"==========================================")

if __name__ == '__main__':
    main()
