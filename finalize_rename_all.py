import os
import sys
import re
import json
import csv
from datetime import datetime
from collections import defaultdict, Counter

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS, IFD
import pillow_heif
import reverse_geocoder as rg
from geopy.geocoders import Nominatim

sys.stdout.reconfigure(encoding='utf-8')
pillow_heif.register_heif_opener()

ROOT_DIR = r"G:\マイドライブ\5-PHOTO\2025"
CACHE_FILE = r"C:\Users\yamachan\.gemini\antigravity-cli\brain\172bb532-7471-4538-b86e-399614703408\scratch\photo_cache.json"
HISTORY_FILE = os.path.join(ROOT_DIR, "rename_history.json")
PREVIEW_CSV = os.path.join(ROOT_DIR, "rename_preview.csv")
ROLLBACK_SCRIPT = os.path.join(ROOT_DIR, "rollback_rename.py")

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.heic', '.png'}

geolocator = Nominatim(user_agent="photo_sorter_2025_finalizer")
geo_cache = {}

def get_location_name(lat, lon):
    if lat is None or lon is None:
        return ""
    cache_key = (round(lat, 3), round(lon, 3))
    if cache_key in geo_cache:
        return geo_cache[cache_key]
    
    # Offline reverse_geocoder first for speed, fallback to Nominatim
    try:
        res = rg.search([(lat, lon)])[0]
        pref = res.get('admin1', '')
        name = res.get('name', '')
        # Map some common romaji to Japanese if possible or keep clean
        loc_str = f"{pref}{name}".strip()
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
                # 1. Date
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
                
                # 2. GPS
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
    
    # Fallbacks for date
    if not date_str:
        parent = os.path.basename(os.path.dirname(file_path))
        m_parent = re.search(r'(202\d{5})', parent)
        if m_parent:
            date_str = m_parent.group(1)
        else:
            fname = os.path.basename(file_path)
            m_fname = re.search(r'(202\d{5})', fname)
            if m_fname:
                date_str = m_fname.group(1)
            else:
                m_fname2 = re.search(r'(\d{2})(\d{2})(\d{2})', fname)
                if m_fname2 and int(m_fname2.group(1)) in [24, 25, 26]:
                    date_str = f"20{m_fname2.group(1)}{m_fname2.group(2)}{m_fname2.group(3)}"
        
        if not date_str:
            try:
                mtime = os.path.getmtime(file_path)
                dt = datetime.fromtimestamp(mtime)
                date_str = dt.strftime("%Y%m%d")
                time_sort = int(dt.strftime("%Y%m%d%H%M%S"))
            except Exception:
                date_str = "20250101"
                
    if time_sort == 0:
        try:
            mtime = os.path.getmtime(file_path)
            time_sort = int(datetime.fromtimestamp(mtime).strftime("%Y%m%d%H%M%S"))
        except Exception:
            time_sort = 0
            
    return date_str, time_sort, lat, lon

def sanitize_filename_part(text):
    if not text:
        return "写真"
    cleaned = re.sub(r'[\\/*?:"<>|\[\]\(\)\s　、。,\.\-\_]+', '', text)
    cleaned = cleaned[:12]
    return cleaned if cleaned else "写真"

def scan_files(root_dir):
    image_files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in VALID_EXTENSIONS:
                # Do not rename files that are scripts or logs
                if f.startswith("batch_rename") or f.startswith("rollback"):
                    continue
                image_files.append(os.path.join(root, f))
    return image_files

def main():
    print("=== Finalizing and Executing All Renames ===")
    
    # 1. Load cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"Loaded cache with {len(cache)} entries.")
        except Exception as e:
            print(f"Error loading cache: {e}")

    # 2. Scan all files
    all_files = scan_files(ROOT_DIR)
    print(f"Total image files found: {len(all_files)}")

    # 3. Group files by directory
    dir_files = defaultdict(list)
    for f in all_files:
        dir_files[os.path.dirname(f)].append(f)

    all_rename_plans = []
    high_quality_count = 0
    fallback_count = 0

    for d, flist in dir_files.items():
        dir_records = []
        gps_locations_in_dir = []

        # Pass 1: EXIF and cached location
        for fpath in flist:
            cached_entry = cache.get(fpath)
            date_str = None
            time_sort = 0
            lat = None
            lon = None
            loc_str = ""
            content = None

            if cached_entry:
                date_str = cached_entry.get("date")
                loc_str = cached_entry.get("location", "")
                content = cached_entry.get("content")

            # Extract EXIF if not cached or incomplete
            if not date_str or not loc_str:
                e_date, e_time, e_lat, e_lon = extract_exif_info(fpath)
                if not date_str:
                    date_str = e_date
                time_sort = e_time
                if not loc_str and e_lat and e_lon:
                    loc_str = get_location_name(e_lat, e_lon)
            else:
                _, time_sort, _, _ = extract_exif_info(fpath)

            if loc_str:
                gps_locations_in_dir.append(loc_str)

            # Determine content
            if content and content != "写真":
                high_quality_count += 1
            else:
                content = "写真"
                fallback_count += 1

            dir_records.append({
                "path": fpath,
                "date": date_str,
                "time_sort": time_sort,
                "location": loc_str,
                "content": content,
                "ext": os.path.splitext(fpath)[1]
            })

        # Infer default location in directory if missing
        default_loc = ""
        if gps_locations_in_dir:
            default_loc = Counter(gps_locations_in_dir).most_common(1)[0][0]

        for item in dir_records:
            if not item["location"]:
                item["location"] = default_loc

        # Sort by capture time
        dir_records.sort(key=lambda x: (x["time_sort"], x["path"]))

        # Assign numbering
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

    print(f"\nPlanned renames: {len(all_rename_plans)} files.")
    print(f"  - High-quality AI content names: {high_quality_count} files")
    print(f"  - Fallback '写真' names: {fallback_count} files")

    # 4. Save Preview CSV
    with open(PREVIEW_CSV, "w", encoding="utf-8-sig", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Directory", "Original Name", "New Name", "Original Path", "New Path"])
        for p in all_rename_plans:
            writer.writerow([p["dir"], p["original_name"], p["new_name"], p["original_path"], p["new_path"]])
    print(f"Preview saved to: {PREVIEW_CSV}")

    # 5. Generate Rollback Script
    with open(ROLLBACK_SCRIPT, "w", encoding="utf-8") as rf:
        rf.write(f'''# Rollback script generated by photo rename tool
import json
import os

HISTORY_FILE = r"{HISTORY_FILE}"

def rollback():
    if not os.path.exists(HISTORY_FILE):
        print(f"History file not found: {{HISTORY_FILE}}")
        return
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)
    print(f"Rolling back {{len(history)}} files...")
    count = 0
    for item in history:
        new_path = item["new_path"]
        orig_path = item["original_path"]
        if os.path.exists(new_path) and new_path != orig_path:
            try:
                os.rename(new_path, orig_path)
                count += 1
            except Exception as e:
                print(f"Error reverting {{new_path}} -> {{orig_path}}: {{e}}")
    print(f"Successfully reverted {{count}} files.")

if __name__ == '__main__':
    rollback()
''')
    print(f"Rollback script generated: {ROLLBACK_SCRIPT}")

    # 6. Execute Rename with temp renaming to avoid collision
    print("\nExecuting actual rename...")
    success_history = []
    renamed_count = 0

    # Two-pass rename to prevent collisions if new_path equals another file's orig_path
    temp_rename_map = []
    for p in all_rename_plans:
        orig = p["original_path"]
        new_p = p["new_path"]
        if orig.lower() == new_p.lower():
            # Same name, skip
            continue
        temp_name = orig + ".tmp_rename"
        try:
            os.rename(orig, temp_name)
            temp_rename_map.append((temp_name, new_p, orig, p))
        except Exception as e:
            print(f"Error moving to temp {orig}: {e}")

    for temp_name, new_p, orig, plan_item in temp_rename_map:
        try:
            # Check if destination exists
            if os.path.exists(new_p):
                # Avoid overwrite
                base, ext = os.path.splitext(new_p)
                new_p = f"{base}_alt{ext}"
                plan_item["new_path"] = new_p
                plan_item["new_name"] = os.path.basename(new_p)
            os.rename(temp_name, new_p)
            success_history.append(plan_item)
            renamed_count += 1
        except Exception as e:
            print(f"Error finalizing rename {temp_name} -> {new_p}: {e}")
            # Try to restore
            try:
                os.rename(temp_name, orig)
            except Exception:
                pass

    with open(HISTORY_FILE, "w", encoding="utf-8") as hf:
        json.dump(success_history, hf, ensure_ascii=False, indent=2)

    print(f"\n==========================================")
    print(f"SUCCESS! Renamed {renamed_count} files.")
    print(f"History recorded in: {HISTORY_FILE}")
    print(f"Rollback script: {ROLLBACK_SCRIPT}")
    print(f"==========================================")

if __name__ == '__main__':
    main()
