@echo off
chcp 65001 > NUL
echo ========================================================
echo  OpenVINO YOLOv11 Ultra-Fast Detection Server (40ms)
echo  Hardware: Intel Core Ultra 7 258V (Intel Arc 140V GPU)
echo ========================================================

echo OpenVINO YOLO 超高速サーバーを起動中 (ポート 8000)...
python openvino_server.py --host 0.0.0.0 --port 8000 --engine yolo --device GPU
pause
