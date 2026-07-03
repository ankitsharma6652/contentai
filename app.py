"""HF Spaces entry point — imports the FastAPI app and runs on port 7860."""
import uvicorn
from content_server import app  # noqa: F401

if __name__ == "__main__":
    uvicorn.run("content_server:app", host="0.0.0.0", port=7860)
