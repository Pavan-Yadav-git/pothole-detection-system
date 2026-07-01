"""
Streamlit App - Pothole Detection System (All-in-One)
Complete inference and frontend in one application
Production-ready for Streamlit Cloud deployment
"""

import streamlit as st
import cv2
import io
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import time
from typing import List, Dict, Any
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent

# Page configuration
st.set_page_config(
    page_title="Pothole Detection",
    page_icon="🕳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration
CONFIG = {
    "model_path": BASE_DIR / "best(1).pt",
    "confidence_threshold": 0.5,
    "max_image_size": 1024,
    "allowed_extensions": {".jpg", ".jpeg", ".png", ".bmp"},
    "max_file_size": 50 * 1024 * 1024,  # 50MB
}


# ============================================================================
# MODEL LOADING AND INFERENCE FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model():
    """Load YOLOv8 model with caching"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.session_state.device = device
        logger.info(f"Using device: {device}")
        
        if not Path(CONFIG["model_path"]).exists():
            st.error(f"❌ Model file not found: {CONFIG['model_path']}")
            return None
        
        model = YOLO(CONFIG["model_path"])
        model.to(device)
        logger.info(f"Model loaded successfully from {CONFIG['model_path']}")
        
        return model
    
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        logger.error(f"Failed to load model: {str(e)}")
        return None


def load_image_bytes(uploaded_file) -> bytes:
    """Return uploaded file contents without consuming the upload stream."""
    if uploaded_file is None:
        return b""
    return uploaded_file.getvalue()


def format_detections(results, image_shape: tuple) -> List[Dict[str, Any]]:
    """Format YOLO results into structured detections"""
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


def detect_potholes_in_image(image_path_or_bytes, model, confidence: float = 0.5) -> Dict[str, Any]:
    """
    Detect potholes in an image
    
    Args:
        image_path_or_bytes: Path to image file or image bytes
        model: YOLO model instance
        confidence: Confidence threshold
    
    Returns:
        Dictionary with detection results
    """
    if model is None:
        return {"success": False, "error": "Model not loaded"}
    
    try:
        start_time = time.time()
        
        # Handle both file paths and bytes
        if isinstance(image_path_or_bytes, bytes):
            if not image_path_or_bytes:
                return {"success": False, "error": "Uploaded image is empty"}

            pil_image = Image.open(io.BytesIO(image_path_or_bytes)).convert("RGB")
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        else:
            image = cv2.imread(str(image_path_or_bytes))
        
        if image is None:
            return {"success": False, "error": "Failed to read image"}
        
        original_shape = image.shape[:2]
        
        # Resize if necessary
        height, width = image.shape[:2]
        if max(height, width) > CONFIG["max_image_size"]:
            scale = CONFIG["max_image_size"] / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Run inference
        results = model.predict(
            image,
            conf=confidence,
            device=st.session_state.get("device", "cpu"),
            verbose=False
        )
        
        # Format results
        detections = format_detections(results, original_shape)
        
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "image_shape": {
                "height": int(original_shape[0]),
                "width": int(original_shape[1]),
                "channels": 3
            },
            "detections": detections,
            "detection_count": len(detections),
            "confidence_threshold": confidence,
            "processing_time": round(processing_time, 3),
            "device": str(st.session_state.get("device", "cpu")),
            "model_version": "1.0.0",
            "timestamp": time.time()
        }
        
        logger.info(f"Detection completed - {len(detections)} detections in {processing_time:.2f}s")
        
        return response
    
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return {"success": False, "error": str(e)}


def draw_detections_on_image(image: Image.Image, detections: List[Dict]) -> np.ndarray:
    """Draw detection boxes on image"""
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    for det in detections:
        bbox = det['bbox']
        x = int(bbox['x'])
        y = int(bbox['y'])
        x2 = x + int(bbox['width'])
        y2 = y + int(bbox['height'])
        confidence = det['confidence']
        
        # Draw rectangle
        cv2.rectangle(img_bgr, (x, y), (x2, y2), (0, 0, 255), 2)
        
        # Put text
        label = f"{det['class']}: {confidence:.2%}"
        cv2.putText(img_bgr, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


# ============================================================================
# STREAMLIT APP UI
# ============================================================================

# Title
st.title("🕳️ Pothole Detection System")
st.markdown("**Developed By: Pavan Yadav**")
st.markdown("---")

# Load model
model = load_model()

if model is None:
    st.error("❌ Failed to load the model. Please check if best(1).pt exists in the directory.")
    st.stop()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Confidence threshold
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # View mode
    view_mode = st.radio(
        "Response View Mode",
        options=["Normal View", "JSON View"],
        help="Choose how to display the detection results"
    )
    
    st.markdown("---")
    st.info("💡 Upload an image to detect potholes using the AI model")

# Main content area
tabs = st.tabs(["Upload & Detect", "Batch Processing", "About"])

with tabs[0]:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image to detect potholes"
        )
        
        if uploaded_file is not None:
            image_bytes = load_image_bytes(uploaded_file)

            # Display uploaded image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(image, caption="Uploaded Image")
            
            # Detection button
            if st.button("🔍 Detect Potholes", use_container_width=True):
                with st.spinner("Processing image..."):
                    try:
                        # Run detection
                        result = detect_potholes_in_image(image_bytes, model, confidence)
                        
                        if result["success"]:
                            # Store result in session state
                            st.session_state.detection_result = result
                            st.session_state.uploaded_image = image
                            st.success("✅ Detection completed!")
                        else:
                            st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.subheader("📊 Results")
        
        if 'detection_result' in st.session_state and st.session_state.detection_result:
            result = st.session_state.detection_result
            
            if view_mode == "Normal View":
                # Display statistics
                detections = result.get('detections', [])
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                
                with col_stats1:
                    st.metric("Total Detections", len(detections))
                
                with col_stats2:
                    if detections:
                        avg_confidence = np.mean([d['confidence'] for d in detections])
                        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                    else:
                        st.metric("Avg Confidence", "N/A")
                
                with col_stats3:
                    st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
                
                # Display detections table
                if detections:
                    st.write("### Detected Potholes:")
                    
                    detection_data = []
                    for i, det in enumerate(detections, 1):
                        detection_data.append({
                            "ID": i,
                            "Confidence": f"{det['confidence']:.2%}",
                            "Class": det['class'],
                            "X": int(det['bbox']['x']),
                            "Y": int(det['bbox']['y']),
                            "Width": int(det['bbox']['width']),
                            "Height": int(det['bbox']['height'])
                        })
                    
                    st.dataframe(detection_data, use_container_width=True)
                else:
                    st.info("✅ No potholes detected in this image!")
                
                # Display annotated image
                if 'uploaded_image' in st.session_state:
                    annotated_img_array = draw_detections_on_image(
                        st.session_state.uploaded_image,
                        detections
                    )
                    annotated_pil = Image.fromarray(annotated_img_array)
                    
                    st.write("### Annotated Image:")
                    st.image(annotated_pil)
            
            else:  # JSON View
                st.write("### JSON Response:")
                st.json(result)
        
        else:
            st.info("Upload an image and click 'Detect Potholes' to see results here")

with tabs[1]:
    st.subheader("📁 Batch Processing")
    st.write("Upload multiple images for batch detection")
    
    uploaded_files = st.file_uploader(
        "Choose multiple images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🔍 Process Batch", use_container_width=True):
        progress_bar = st.progress(0)
        results_container = st.container()
        
        batch_results = []
        
        for idx, file in enumerate(uploaded_files):
            try:
                image_bytes = load_image_bytes(file)
                Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                result = detect_potholes_in_image(image_bytes, model, confidence)
                
                if result.get("success"):
                    batch_results.append({
                        'filename': file.name,
                        'detections': result.get('detection_count', 0),
                        'status': 'Success',
                        'time': result.get('processing_time', 0)
                    })
                else:
                    batch_results.append({
                        'filename': file.name,
                        'detections': 0,
                        'status': 'Failed',
                        'time': 0
                    })
            
            except Exception as e:
                batch_results.append({
                    'filename': file.name,
                    'detections': 0,
                    'status': 'Error',
                    'time': 0
                })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # Display results
        st.success("✅ Batch processing completed!")
        results_df = __import__('pandas').DataFrame(batch_results)
        st.dataframe(results_df, use_container_width=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results",
            csv,
            "batch_results.csv",
            "text/csv"
        )

with tabs[2]:
    st.subheader("ℹ️ About")
    st.markdown("""
    ### Pothole Detection System
    
    This application uses advanced computer vision model to detect potholes in road images.
    
    **Features:**
    - 📤 Upload single or multiple images
    - 🔍 Real-time detection with confidence scores
    - 📊 Detailed statistics and visualization
    - 📋 Batch processing support
    - 💾 Exportable results in CSV and JSON formats
    
    **How it works:**
    1. Upload an image containing a road
    2. The AI model analyzes the image
    3. Detects and localizes potholes with bounding boxes
    4. Returns confidence scores for each detection
    
    **Technologies:**
    - Frontend: Streamlit
    - Model: YOLO26 (fine-tuned for pothole detection)
    - Computer Vision: OpenCV
    - Deep Learning: PyTorch
    
    **Response Formats:**
    - **Normal View**: Displays statistics, table, and annotated images
    - **JSON View**: Shows the complete response in JSON format
    
    ---
    **Version:** 1.0.0  
    **Status:** Production Ready ✅  
    **Deployment:** Streamlit Cloud
    """)
    
    st.markdown("---")
    st.caption("👨‍💻 Developed By: Pavan Yadav")
