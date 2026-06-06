# Pothole Detection System

This folder now contains only the files needed for the Streamlit pothole detection app.

## Included Files

- `streamlit_app.py`: Streamlit UI and YOLO inference logic
- `best(1).pt`: trained pothole detection model
- `requirements.txt`: Python dependencies
- `runtime.txt`: Python version for deployment
- `.streamlit/config.toml`: Streamlit configuration
- `.gitignore`: local Git ignore rules

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Project Structure

```text
pothole_detect_system/
├── .streamlit/
│   └── config.toml
├── best(1).pt
├── README.md
├── requirements.txt
├── runtime.txt
└── streamlit_app.py
```
