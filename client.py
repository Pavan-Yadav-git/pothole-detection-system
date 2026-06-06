"""
Python Client Examples for Pathhole Detection API

This file demonstrates various ways to interact with the API
"""

import requests
import json
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
from PIL import Image
import io


class PathholeClient:
    """
    Client for Pathhole Detection API
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if API is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get API information"""
        response = self.session.get(f"{self.base_url}/info")
        return response.json()
    
    def detect_image(self, image_path: str, confidence: float = 0.5) -> Dict[str, Any]:
        """
        Detect potholes in an image
        
        Args:
            image_path: Path to image file
            confidence: Confidence threshold (0-1)
        
        Returns:
            Detection results
        """
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {'confidence': confidence}
            response = self.session.post(
                f"{self.base_url}/detect",
                files=files,
                params=params
            )
        
        return response.json()
    
    def detect_from_bytes(self, image_bytes: bytes, confidence: float = 0.5) -> Dict[str, Any]:
        """
        Detect potholes from image bytes
        
        Args:
            image_bytes: Image data as bytes
            confidence: Confidence threshold
        
        Returns:
            Detection results
        """
        files = {'file': ('image.jpg', io.BytesIO(image_bytes), 'image/jpeg')}
        params = {'confidence': confidence}
        response = self.session.post(
            f"{self.base_url}/detect",
            files=files,
            params=params
        )
        
        return response.json()
    
    def detect_from_url(self, image_url: str, confidence: float = 0.5) -> Dict[str, Any]:
        """
        Detect potholes from image URL
        
        Args:
            image_url: URL to image
            confidence: Confidence threshold
        
        Returns:
            Detection results
        """
        img_response = requests.get(image_url)
        image_bytes = img_response.content
        return self.detect_from_bytes(image_bytes, confidence)
    
    def batch_detect(self, image_paths: List[str], confidence: float = 0.5) -> Dict[str, Any]:
        """
        Detect potholes in multiple images
        
        Args:
            image_paths: List of image paths
            confidence: Confidence threshold
        
        Returns:
            Batch detection results
        """
        files = []
        for path in image_paths:
            with open(path, 'rb') as f:
                files.append(('files', (Path(path).name, f, 'image/jpeg')))
        
        params = {'confidence': confidence}
        response = self.session.post(
            f"{self.base_url}/batch-detect",
            files=files,
            params=params
        )
        
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """Get available models"""
        response = self.session.get(f"{self.base_url}/models")
        return response.json()
    
    def draw_detections(self, image_path: str, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes on image
        
        Args:
            image_path: Path to image
            detections: List of detection results
        
        Returns:
            Annotated image as numpy array
        """
        image = cv2.imread(image_path)
        
        for detection in detections:
            bbox = detection['bbox']
            x = int(bbox['x'])
            y = int(bbox['y'])
            x2 = x + int(bbox['width'])
            y2 = y + int(bbox['height'])
            confidence = detection['confidence']
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 2)
            
            # Put text
            label = f"{detection['class']}: {confidence:.2%}"
            cv2.putText(image, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return image
    
    def save_annotated_image(self, image_path: str, detections: List[Dict], 
                           output_path: str):
        """
        Save annotated image
        
        Args:
            image_path: Input image path
            detections: Detection results
            output_path: Output image path
        """
        annotated = self.draw_detections(image_path, detections)
        cv2.imwrite(output_path, annotated)
    
    def export_results_json(self, results: Dict, output_path: str):
        """
        Export results to JSON
        
        Args:
            results: Detection results
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def export_results_csv(self, results: Dict, output_path: str):
        """
        Export results to CSV
        
        Args:
            results: Detection results
            output_path: Output file path
        """
        import csv
        
        with open(output_path, 'w', newline='') as f:
            if results.get('detections'):
                fieldnames = ['class', 'confidence', 'x', 'y', 'width', 'height']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for detection in results['detections']:
                    bbox = detection['bbox']
                    writer.writerow({
                        'class': detection['class'],
                        'confidence': detection['confidence'],
                        'x': bbox['x'],
                        'y': bbox['y'],
                        'width': bbox['width'],
                        'height': bbox['height']
                    })


# Example usage functions

def example_single_image():
    """Example: Detect in single image"""
    print("Example 1: Single Image Detection")
    print("-" * 50)
    
    client = PathholeClient()
    
    # Check health
    if not client.health_check():
        print("❌ API is not running")
        return
    
    # Detect
    image_path = "path/to/your/image.jpg"
    results = client.detect_image(image_path, confidence=0.5)
    
    print(f"✅ Detections: {results['detection_count']}")
    print(json.dumps(results, indent=2))
    
    # Save annotated image
    client.save_annotated_image(image_path, results['detections'], "output_annotated.jpg")


def example_batch_processing():
    """Example: Batch processing"""
    print("\nExample 2: Batch Processing")
    print("-" * 50)
    
    client = PathholeClient()
    
    image_paths = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg"
    ]
    
    results = client.batch_detect(image_paths, confidence=0.5)
    
    print(f"✅ Processed {results['processed_files']} of {results['total_files']} files")
    print(json.dumps(results, indent=2))


def example_from_url():
    """Example: Detect from URL"""
    print("\nExample 3: Detect from URL")
    print("-" * 50)
    
    client = PathholeClient()
    
    image_url = "https://example.com/road_image.jpg"
    results = client.detect_from_url(image_url, confidence=0.6)
    
    print(f"✅ Detections: {results['detection_count']}")


def example_export_results():
    """Example: Export results"""
    print("\nExample 4: Export Results")
    print("-" * 50)
    
    client = PathholeClient()
    
    results = client.detect_image("image.jpg", confidence=0.5)
    
    # Export to JSON
    client.export_results_json(results, "results.json")
    print("✅ Exported to results.json")
    
    # Export to CSV
    client.export_results_csv(results, "results.csv")
    print("✅ Exported to results.csv")


def example_get_api_info():
    """Example: Get API information"""
    print("\nExample 5: API Information")
    print("-" * 50)
    
    client = PathholeClient()
    
    info = client.get_info()
    print(f"API Status: {info['status']}")
    print(f"Device: {info['device']}")
    print(f"CUDA Available: {info['cuda_available']}")
    
    models = client.list_models()
    print(f"Available Models: {models['available_models']}")


if __name__ == "__main__":
    # Run examples
    # Uncomment to run:
    
    # example_single_image()
    # example_batch_processing()
    # example_from_url()
    # example_export_results()
    # example_get_api_info()
    
    # Or create a simple interactive session:
    client = PathholeClient()
    
    print("🕳️  Pathhole Detection API Client")
    print("="*50)
    
    if client.health_check():
        print("✅ API is healthy")
        info = client.get_info()
        print(f"   Device: {info['device']}")
        print(f"   Model: {info['model_name']}")
    else:
        print("❌ API is not responding")
