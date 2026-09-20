import sys
import traceback

print("=== OpenVINO VLM 環境診断スクリプト ===")
print("Python バージョン:", sys.version)

print("\n[1] OpenVINO の確認:")
try:
    import openvino as ov
    print("  -> OK: OpenVINO バージョン:", ov.__version__)
    core = ov.Core()
    devices = core.available_devices
    print("  -> 利用可能デバイス:", devices)
except Exception as e:
    print("  -> NG:", e)

print("\n[2] openvino-genai の確認:")
try:
    import openvino_genai as ov_genai
    print("  -> OK: openvino_genai バージョン:", ov_genai.__version__)
except Exception as e:
    print("  -> NG: openvino_genai のインポートに失敗しました:")
    print("     ", e)
    print("     解決策: pip install openvino-genai huggingface_hub を実行してください。")

print("\n[3] huggingface_hub の確認:")
try:
    import huggingface_hub as hf_hub
    print("  -> OK: huggingface_hub バージョン:", hf_hub.__version__)
except Exception as e:
    print("  -> NG:", e)

print("\n[4] Qwen2.5-VL モデルのダウンロードと GPU ロードテスト:")
try:
    import openvino_genai as ov_genai
    from huggingface_hub import snapshot_download
    
    model_id = "OpenVINO/Qwen2.5-VL-7B-Instruct-int4-ov"
    print(f"  -> モデル '{model_id}' のダウンロード確認中...")
    model_path = snapshot_download(repo_id=model_id)
    print(f"  -> モデルパス: {model_path}")
    
    print("  -> Intel Arc GPU へのロードを試行中...")
    pipe = ov_genai.VLMPipeline(model_path, "GPU")
    print("  -> 成功！ Intel Arc GPU に Qwen2.5-VL がロードされました！")
except Exception as e:
    print("  -> エラーが発生しました:")
    traceback.print_exc()

print("\n==========================================")
