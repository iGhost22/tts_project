import os
import sys
import uvicorn

# Add the project root to the Python path to find modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Create result directory if it doesn't exist
result_dir = os.path.join(project_root, "result")
os.makedirs(result_dir, exist_ok=True)

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        reload_dirs=[os.path.dirname(os.path.abspath(__file__))],
    )
