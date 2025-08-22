#!/usr/bin/env python3
"""
Quick test script for deployed RunPod endpoint
Usage: python quick_test.py YOUR_ENDPOINT_URL YOUR_API_KEY
"""

import sys
import json
import requests
import time

def test_endpoint(endpoint_url, api_key):
    """Test the deployed RunPod endpoint."""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    test_payload = {
        "input": {
            "text": "Hello! This is a quick test of your Higgs Audio endpoint on RunPod.",
            "temperature": 0.7,
            "max_new_tokens": 128,
            "output_format": "wav"
        }
    }
    
    print(f"🚀 Testing endpoint: {endpoint_url}")
    print(f"📝 Test text: {test_payload['input']['text']}")
    print("⏳ Sending request...")
    
    start_time = time.time()
    
    try:
        response = requests.post(endpoint_url, headers=headers, json=test_payload, timeout=120)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"⏱️  Response time: {total_time:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if "error" in result:
                print(f"❌ API Error: {result['error']}")
                return False
            else:
                print("✅ Success!")
                print(f"  - Audio duration: {result.get('duration', 'N/A')} seconds")
                print(f"  - Sampling rate: {result.get('sampling_rate', 'N/A')} Hz")
                print(f"  - Generated text: {result.get('generated_text', 'N/A')[:100]}...")
                
                if "timing" in result:
                    print(f"  - Generation time: {result['timing']['generation_time']} seconds")
                
                # Save audio file
                if result.get("audio_base64"):
                    import base64
                    audio_data = base64.b64decode(result["audio_base64"])
                    with open("endpoint_test_output.wav", "wb") as f:
                        f.write(audio_data)
                    print("  - Audio saved to: endpoint_test_output.wav")
                
                return True
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>120 seconds)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python quick_test.py YOUR_ENDPOINT_URL YOUR_API_KEY")
        print()
        print("Example:")
        print("python quick_test.py https://api.runpod.ai/v2/abc123/run your_api_key_here")
        sys.exit(1)
    
    endpoint_url = sys.argv[1]
    api_key = sys.argv[2]
    
    success = test_endpoint(endpoint_url, api_key)
    
    if success:
        print("\n🎉 Your Higgs Audio endpoint is working correctly!")
    else:
        print("\n💥 There was an issue with your endpoint. Check the logs in RunPod.")

if __name__ == "__main__":
    main()
