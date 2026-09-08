import os
import sys

# Add backend directory to python path
backend_dir = os.path.join(os.path.dirname(__file__), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from main import app
import gradio as gr

# Create a Gradio interface wrapper for HuggingFace Spaces
with gr.Blocks(title="Operation Safe Vault Backend") as demo:
    gr.Markdown("# 🛡️ Operation Safe Vault — API Backend Server")
    gr.Markdown("FastAPI server active with PyTorch Spectrogram CNN, Wav2Vec2 Deepfake Detector, and Multi-Modal Threat Fusion Engine.")

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
