"""
app.py - Main entrypoint for Hugging Face Spaces and Cloud Deployment
"""

from advanced_dataset_loader import preload_all_models
from app_advanced_master import CUSTOM_CSS, THEME, build_interface

print("[*] Preloading models for Hugging Face Spaces...")
preload_all_models()

demo = build_interface()

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS)
