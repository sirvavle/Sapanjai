import os
from huggingface_hub import hf_hub_download

HF_TOKEN = os.getenv("HF_TOKEN")

print("📦 Preloading emotion model...")
hf_hub_download(
    repo_id="Patzamangajuice/best_goemotions_mode",
    filename="best_goemotions_model.pt",
    token=HF_TOKEN,
    cache_dir="/app/models"
)

print("📦 Preloading zero-shot model (config only)...")
hf_hub_download(
    repo_id="facebook/bart-large-mnli",
    filename="config.json",
    cache_dir="/app/models"
)

print("✅ Models downloaded and cached.")
