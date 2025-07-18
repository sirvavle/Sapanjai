from huggingface_hub import hf_hub_download
import os

# Override HF home to avoid /app permission errors
os.environ["HF_HOME"] = "/tmp"

HF_TOKEN = os.getenv("HF_TOKEN")

model_path = hf_hub_download(
    repo_id="Patzamangajuice/best_goemotions_mode",
    filename="best_goemotions_model.pt",
    token=HF_TOKEN,
    cache_dir="/tmp/models"
)

print(f"✅ Emotion model preloaded to {model_path}")
