"""
Streamlit Frontend for Pothole Detection
"""

import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import json
from io import BytesIO
import base64

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

# Title
st.title("🕳️ Pothole Detection System")
st.markdown("**Developed By: Pavan Yadav**")
st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API endpoint
    api_url = st.text_input(
        "API Endpoint",
        value="http://localhost:8000",
        help="URL of the FastAPI server"
    )
    
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
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")
            
            # Detection button
            if st.button("🔍 Detect Potholes", use_container_width=True):
                with st.spinner("Processing image..."):
                    try:
                        # Prepare image for API
                        image_array = np.array(image)
                        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
                        image_bytes = buffer.tobytes()
                        
                        # Send to API
                        files = {'file': ('image.jpg', BytesIO(image_bytes), 'image/jpeg')}
                        params = {'confidence': confidence}
                        
                        response = requests.post(
                            f"{api_url}/detect",
                            files=files,
                            params=params,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Store result in session state
                            st.session_state.detection_result = result
                            st.session_state.uploaded_image = image
                            st.success("✅ Detection completed!")
                        else:
                            st.error(f"❌ API Error: {response.status_code}")
                            st.error(response.text)
                    
                    except requests.exceptions.ConnectionError:
                        st.error(f"❌ Cannot connect to API at {api_url}")
                        st.info("Make sure the FastAPI server is running on the specified endpoint")
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
                    annotated_img = st.session_state.uploaded_image.copy()
                    
                    # Draw bounding boxes
                    img_array = np.array(annotated_img)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    for det in detections:
                        bbox = det['bbox']
                        x = int(bbox['x'])
                        y = int(bbox['y'])
                        width = int(bbox['width'])
                        height = int(bbox['height'])
                        confidence = det['confidence']
                        
                        # Draw rectangle
                        cv2.rectangle(img_bgr, (x, y), (x + width, y + height), (0, 0, 255), 2)
                        
                        # Put text
                        label = f"{det['class']}: {confidence:.2%}"
                        cv2.putText(img_bgr, label, (x, y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    annotated_pil = Image.fromarray(img_rgb)
                    
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
                image = Image.open(file)
                image_array = np.array(image)
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
                image_bytes = buffer.tobytes()
                
                files = {'file': ('image.jpg', BytesIO(image_bytes), 'image/jpeg')}
                params = {'confidence': confidence}
                
                response = requests.post(
                    f"{api_url}/detect",
                    files=files,
                    params=params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    batch_results.append({
                        'filename': file.name,
                        'detections': len(result.get('detections', [])),
                        'status': 'Success'
                    })
                else:
                    batch_results.append({
                        'filename': file.name,
                        'detections': 0,
                        'status': 'Failed'
                    })
            
            except Exception as e:
                batch_results.append({
                    'filename': file.name,
                    'detections': 0,
                    'status': f'Error: {str(e)}'
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
    ### Pathhole Detection System
    
    This application uses advanced deep learning models to detect potholes in road images.
    
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
    - Backend: FastAPI
    - Frontend: Streamlit
    - Model: YOLOv26n (fine-tuned for pothole detection)
    - Computer Vision: OpenCV
    
    **Response Formats:**
    - **Normal View**: Displays statistics, table, and annotated images
    - **JSON View**: Shows the complete API response in JSON format
    
    ---
    **Version:** 1.0.0  
    **Status:** Production Ready ✅
    """)
    
    st.markdown("---")
    st.info("📝 For API documentation, visit `/docs` endpoint on the FastAPI server")
    
    st.markdown("---")
    st.caption("👨‍💻 Developed By: Pavan Yadav")
