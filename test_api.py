"""
Test script for Pathhole Detection API
Run this to verify the system is working correctly
"""

import requests
import json
import time
from pathlib import Path
import sys

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_IMAGE_PATH = "./test_image.jpg"

def test_health():
    """Test health endpoint"""
    print("\n📋 Testing Health Endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_info():
    """Test info endpoint"""
    print("\n📋 Testing Info Endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_detection(image_path: str, confidence: float = 0.5):
    """Test detection endpoint"""
    print(f"\n📋 Testing Detection Endpoint...")
    print(f"Image: {image_path}")
    print(f"Confidence: {confidence}")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as img_file:
            files = {'file': img_file}
            params = {'confidence': confidence}
            
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/detect",
                files=files,
                params=params
            )
            elapsed = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Processing Time: {elapsed:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Detection Results:")
            print(f"  Detections: {result.get('detection_count', 0)}")
            print(f"  Device: {result.get('device', 'N/A')}")
            print(f"  Model Version: {result.get('model_version', 'N/A')}")
            
            if result.get('detections'):
                print(f"\n  Found {len(result['detections'])} pothole(s):")
                for i, det in enumerate(result['detections'], 1):
                    print(f"    {i}. Confidence: {det['confidence']:.2%}, "
                          f"Class: {det['class']}, "
                          f"BBox: ({det['bbox']['x']}, {det['bbox']['y']}, "
                          f"{det['bbox']['width']}, {det['bbox']['height']})")
            else:
                print(f"  ✅ No potholes detected")
            
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_models_list():
    """Test models list endpoint"""
    print("\n📋 Testing Models List Endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_root():
    """Test root endpoint"""
    print("\n📋 Testing Root Endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_all_tests(image_path: str = TEST_IMAGE_PATH):
    """Run all tests"""
    print("="*60)
    print("🧪 Pathhole Detection API - Test Suite")
    print("="*60)
    
    results = {
        "root": test_root(),
        "health": test_health(),
        "info": test_info(),
        "models": test_models_list(),
    }
    
    if Path(image_path).exists():
        results["detection"] = test_detection(image_path)
        results["detection_low_conf"] = test_detection(image_path, confidence=0.3)
        results["detection_high_conf"] = test_detection(image_path, confidence=0.9)
    else:
        print(f"\n⚠️  Skipping detection tests - no test image found at {image_path}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        return False


if __name__ == "__main__":
    # Check if API is running
    print(f"\nConnecting to API at {API_BASE_URL}...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print("✅ API is running\n")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API at {API_BASE_URL}")
        print("\nMake sure the FastAPI server is running:")
        print(f"  python main.py\n")
        sys.exit(1)
    
    # Get image path from command line if provided
    image_path = sys.argv[1] if len(sys.argv) > 1 else TEST_IMAGE_PATH
    
    # Run tests
    success = run_all_tests(image_path)
    sys.exit(0 if success else 1)
