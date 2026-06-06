"""
FastAPI Backend for Pathhole Detection
Production-ready inference server
"""

import os
import logging
import time
from typing import List, Dict, Any
from pathlib import Path
import tempfile
import shutil

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pathhole Detection API",
    description="Production-ready API for detecting potholes in road images",
    version="1.0.0"
)

# Add CORS middleware for Vercel deployment and local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
device = None

# Configuration
CONFIG = {
    "model_path": "./best(1).pt",
    "confidence_threshold": 0.5,
    "max_image_size": 1024,
    "allowed_extensions": {".jpg", ".jpeg", ".png", ".bmp"},
    "max_file_size": 50 * 1024 * 1024,  # 50MB
}


def load_model():
    """Load YOLOv8 model"""
    global model, device
    
    try:
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model
        if not os.path.exists(CONFIG["model_path"]):
            raise FileNotFoundError(f"Model file not found: {CONFIG['model_path']}")
        
        model = YOLO(CONFIG["model_path"])
        model.to(device)
        logger.info(f"Model loaded successfully from {CONFIG['model_path']}")
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False


def process_image(image_data: bytes, max_size: int = CONFIG["max_image_size"]) -> np.ndarray:
    """
    Process image data and prepare for inference
    
    Args:
        image_data: Raw image bytes
        max_size: Maximum image dimension
    
    Returns:
        Processed image array
    """
    # Read image from bytes
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    # Resize if necessary
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
    
    return img


def format_detections(results, image_shape: tuple) -> List[Dict[str, Any]]:
    """
    Format YOLO results into structured detections
    
    Args:
        results: YOLO inference results
        image_shape: Original image shape (height, width)
    
    Returns:
        List of detection dictionaries
    """
    detections = []
    
    if results is None or len(results) == 0:
        return detections
    
    result = results[0]
    
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    
    # Extract boxes and confidences
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    class_names = result.names
    
    for box, confidence, class_id in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = box.astype(int)
        width = x2 - x1
        height = y2 - y1
        
        detection = {
            "class": class_names.get(class_id, f"class_{class_id}"),
            "class_id": int(class_id),
            "confidence": float(confidence),
            "bbox": {
                "x": int(x1),
                "y": int(y1),
                "width": int(width),
                "height": int(height),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            }
        }
        detections.append(detection)
    
    return detections


def validate_image_file(filename: str, file_size: int) -> bool:
    """Validate image file"""
    ext = Path(filename).suffix.lower()
    
    if ext not in CONFIG["allowed_extensions"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {CONFIG['allowed_extensions']}"
        )
    
    if file_size > CONFIG["max_file_size"]:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {CONFIG['max_file_size'] / 1024 / 1024:.1f}MB"
        )
    
    return True


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting up application...")
    if not load_model():
        logger.error("Failed to load model on startup")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Pathhole Detection API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "detect": "POST /detect",
            "health": "GET /health",
            "info": "GET /info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "not initialized",
        "timestamp": time.time()
    }


@app.get("/info")
async def get_info():
    """Get API and model information"""
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    
    return {
        "status": "running",
        "model_name": model.model_name,
        "model_path": CONFIG["model_path"],
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "max_image_size": CONFIG["max_image_size"],
        "max_file_size": CONFIG["max_file_size"],
        "allowed_extensions": list(CONFIG["allowed_extensions"]),
        "version": "1.0.0"
    }


@app.post("/detect")
async def detect_potholes(
    file: UploadFile = File(...),
    confidence: float = Query(0.5, ge=0.0, le=1.0, description="Confidence threshold")
):
    """
    Detect potholes in uploaded image
    
    Args:
        file: Image file
        confidence: Confidence threshold for detections
    
    Returns:
        JSON with detections and metadata
    """
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    temp_dir = None
    
    try:
        start_time = time.time()
        
        # Validate file
        file_content = await file.read()
        validate_image_file(file.filename, len(file_content))
        
        # Save temporarily for YOLO
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_path, 'wb') as f:
            f.write(file_content)
        
        # Process image
        image = cv2.imread(temp_path)
        original_shape = image.shape[:2]
        
        # Run inference
        results = model.predict(
            temp_path,
            conf=confidence,
            device=device,
            verbose=False
        )
        
        # Format results
        detections = format_detections(results, original_shape)
        
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "filename": file.filename,
            "image_shape": {
                "height": int(original_shape[0]),
                "width": int(original_shape[1]),
                "channels": 3
            },
            "detections": detections,
            "detection_count": len(detections),
            "confidence_threshold": confidence,
            "processing_time": round(processing_time, 3),
            "device": str(device),
            "model_version": "1.0.0",
            "timestamp": time.time()
        }
        
        logger.info(f"Detection completed: {file.filename} - {len(detections)} detections in {processing_time:.2f}s")
        
        return JSONResponse(content=response, status_code=200)
    
    except HTTPException as e:
        logger.error(f"Validation error: {e.detail}")
        raise e
    
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


@app.post("/batch-detect")
async def batch_detect(
    files: List[UploadFile] = File(...),
    confidence: float = Query(0.5, ge=0.0, le=1.0)
):
    """
    Batch detect potholes in multiple images
    
    Args:
        files: List of image files
        confidence: Confidence threshold
    
    Returns:
        List of detection results
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images per batch")
    
    results = []
    
    for file in files:
        try:
            file_content = await file.read()
            validate_image_file(file.filename, len(file_content))
            
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file.filename)
            
            with open(temp_path, 'wb') as f:
                f.write(file_content)
            
            image = cv2.imread(temp_path)
            detection_results = model.predict(
                temp_path,
                conf=confidence,
                device=device,
                verbose=False
            )
            
            detections = format_detections(detection_results, image.shape[:2])
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "detection_count": len(detections),
                "detections": detections
            })
            
            shutil.rmtree(temp_dir)
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_files": len(files),
        "processed_files": len([r for r in results if r['status'] == 'success']),
        "results": results,
        "timestamp": time.time()
    }


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": ["best(1).pt"],
        "current_model": CONFIG["model_path"],
        "model_info": {
            "type": "YOLOv8",
            "task": "object_detection",
            "classes": ["pothole"]
        }
    }


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        workers=1
    )
