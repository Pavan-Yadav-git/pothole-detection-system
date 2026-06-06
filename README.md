# 🕳️ Pathhole Detection System

A production-ready deep learning application for detecting potholes in road images using FastAPI backend and Streamlit frontend.

## Features

✅ **Real-time Detection**: Advanced YOLOv8 model for accurate pothole detection  
✅ **Single & Batch Processing**: Upload one or multiple images  
✅ **Multiple View Modes**: Normal and JSON response formats  
✅ **Production Ready**: Optimized for deployment  
✅ **Vercel Ready**: Easy cloud deployment  
✅ **RESTful API**: Complete API documentation with Swagger UI  
✅ **Docker Support**: Containerized for consistent environments  
✅ **CORS Enabled**: Frontend-friendly configuration  

## Quick Start

### Local Development Setup

#### 1. Prerequisites
- Python 3.9+
- pip or conda
- Git

#### 2. Installation

```bash
# Clone or download the project
cd pathhole_detect

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Run Locally

**Terminal 1 - Start FastAPI Backend:**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Detect Endpoint**: POST http://localhost:8000/detect

**Terminal 2 - Start Streamlit Frontend:**
```bash
streamlit run app.py
```

The frontend will be available at `http://localhost:8501`

## API Documentation

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "timestamp": 1699564800
}
```

#### 2. Get Model Info
```http
GET /info
```

**Response:**
```json
{
  "status": "running",
  "model_name": "YOLOv8",
  "device": "cuda",
  "cuda_available": true,
  "version": "1.0.0"
}
```

#### 3. Detect Potholes
```http
POST /detect
```

**Parameters:**
- `file` (FormData): Image file (jpg, jpeg, png, bmp)
- `confidence` (query): Confidence threshold (0.0-1.0, default: 0.5)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/detect?confidence=0.5" \
  -H "accept: application/json" \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "success": true,
  "filename": "image.jpg",
  "image_shape": {
    "height": 720,
    "width": 1280,
    "channels": 3
  },
  "detections": [
    {
      "class": "pothole",
      "class_id": 0,
      "confidence": 0.92,
      "bbox": {
        "x": 100,
        "y": 150,
        "width": 80,
        "height": 60,
        "x1": 100,
        "y1": 150,
        "x2": 180,
        "y2": 210
      }
    }
  ],
  "detection_count": 1,
  "confidence_threshold": 0.5,
  "processing_time": 0.234,
  "device": "cuda:0",
  "model_version": "1.0.0",
  "timestamp": 1699564800
}
```

#### 4. Batch Detection
```http
POST /batch-detect
```

**Parameters:**
- `files` (FormData): Multiple image files
- `confidence` (query): Confidence threshold (default: 0.5)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/batch-detect?confidence=0.5" \
  -H "accept: application/json" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

## Deployment Guide

### Deploy to Vercel

Vercel is the easiest platform to deploy this application.

#### Prerequisites:
- Vercel account (free)
- GitHub account
- Git installed locally

#### Steps:

1. **Prepare Your Repository**
   ```bash
   # Ensure all files are in your repo:
   # - main.py
   # - app.py
   # - requirements.txt
   # - best(1).pt
   # - vercel.json
   # - .env.example
   ```

2. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Pathhole detection system"
   git branch -M main
   git remote add origin https://github.com/your-username/pathhole-detection.git
   git push -u origin main
   ```

3. **Deploy to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Configure environment variables in Project Settings
   - Set custom domains if needed
   - Click "Deploy"

4. **Configure Environment**
   - In Vercel dashboard, go to Settings → Environment Variables
   - Add the following:
     ```
     MODEL_PATH=./best(1).pt
     API_PORT=8000
     ```

#### Important Notes for Vercel:
- Model file (best(1).pt) needs to be included in the repository
- Keep model size under 100MB for optimal performance
- Use serverless functions if available

### Deploy with Docker (Recommended for production)

#### 1. Build Docker Image
```bash
docker build -t pathhole-detection:latest .
```

#### 2. Run Container
```bash
docker run -p 8000:8000 \
  -e API_HOST=0.0.0.0 \
  -e API_PORT=8000 \
  pathhole-detection:latest
```

#### 3. Using Docker Compose (For full stack)
```bash
docker-compose up -d
```

This will start both the API (port 8000) and Streamlit frontend (port 8501).

### Deploy to Heroku

```bash
# Install Heroku CLI
brew tap heroku/brew && brew install heroku

# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Add buildpack
heroku buildpacks:add heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

### Deploy to AWS / Google Cloud / Azure

All platforms support Python applications. Refer to their respective documentation:
- AWS EC2, ECS, Lambda
- Google Cloud Run, App Engine
- Azure Container Instances, App Service

## Usage

### Streamlit Frontend Features

1. **Single Image Detection**
   - Upload image
   - Adjust confidence threshold
   - View results in Normal or JSON format
   - Download annotated image

2. **Batch Processing**
   - Upload multiple images
   - Process all at once
   - Export results as CSV

3. **Statistics**
   - Total detections count
   - Average confidence scores
   - Processing time per image
   - Image metadata

### Customization

#### Change Model Confidence Threshold
Edit in Streamlit sidebar or pass as parameter to API:
```bash
curl -X POST "http://localhost:8000/detect?confidence=0.7" \
  -F "file=@image.jpg"
```

#### Adjust Image Size Limits
Modify in `main.py`:
```python
CONFIG = {
    "max_image_size": 2048,  # Increase for larger images
    "max_file_size": 100 * 1024 * 1024,  # 100MB
}
```

#### Use Different Model
Replace `best(1).pt` with your model and update CONFIG in `main.py`:
```python
CONFIG = {
    "model_path": "./your_model.pt",
}
```

## Troubleshooting

### Model Not Loading
```
Error: Failed to load model: Model file not found
```
**Solution**: Ensure `best(1).pt` is in the project root directory

### CUDA Not Available (GPU not detected)
```python
# Falls back to CPU automatically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Port Already in Use
```bash
# Change port in main.py or run with:
python main.py --port 8001
```

### Memory Issues
Reduce `max_image_size` in CONFIG:
```python
CONFIG = {
    "max_image_size": 512,  # Smaller images use less memory
}
```

## Performance Tips

1. **GPU Acceleration**: Install CUDA for 10-20x faster inference
2. **Batch Processing**: Process multiple images together
3. **Image Preprocessing**: Resize large images before upload
4. **Caching**: Results are cached by default
5. **Load Balancing**: Use multiple workers in production

## Security Considerations

✅ File type validation  
✅ File size limits  
✅ Input sanitization  
✅ CORS protection  
✅ Rate limiting ready (can be added)  
✅ Authentication ready (can be added with JWT)  

## API Response Formats

### Success Response
```json
{
  "success": true,
  "filename": "image.jpg",
  "detections": [...],
  "detection_count": 5,
  "processing_time": 0.45
}
```

### Error Response
```json
{
  "detail": "Invalid file type. Allowed: {'.jpg', '.jpeg', '.png', '.bmp'}"
}
```

## System Requirements

**Minimum:**
- 4GB RAM
- 2GB disk space (without model)
- Python 3.9+

**Recommended (for production):**
- 16GB RAM
- 8GB disk space
- GPU (NVIDIA with CUDA support)
- Python 3.10+

## File Structure

```
pathhole_detect/
├── main.py                    # FastAPI backend
├── app.py                     # Streamlit frontend
├── requirements.txt           # Python dependencies
├── best(1).pt                # YOLOv8 model
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose setup
├── vercel.json              # Vercel configuration
├── .env.example             # Environment template
├── README.md                # This file
└── PotholeDetection_data/   # Training data (optional)
```

## Environment Variables

Create a `.env` file:
```
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=./best(1).pt
CONFIDENCE_THRESHOLD=0.5
MAX_IMAGE_SIZE=1024
LOG_LEVEL=INFO
ENVIRONMENT=production
```

## License

This project is provided as-is for research and commercial use.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs` endpoint
3. Check application logs
4. Verify all files are present

## Version History

- **v1.0.0** (Current)
  - Initial release
  - YOLOv8 model integration
  - FastAPI backend
  - Streamlit frontend
  - Batch processing
  - Multiple response formats
  - Vercel deployment ready

## Future Enhancements

- [ ] Real-time video stream processing
- [ ] Advanced filtering options
- [ ] Analytics dashboard
- [ ] User authentication
- [ ] Database integration
- [ ] Model fine-tuning interface
- [ ] Mobile app
- [ ] Multi-model support

---


