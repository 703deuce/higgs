#!/usr/bin/env python3
"""
Test script to verify the new API structure with user_id parameter
"""

import requests
import json
import time
import base64

# RunPod endpoint configuration
RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/zeqk8y61qusvji/run"
API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"

def poll_job_status(job_id, test_name):
    """Poll job status until completion"""
    status_url = f"https://api.runpod.ai/v2/zeqk8y61qusvji/status/{job_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    print(f"  🔄 Polling job status: {job_id}")
    
    while True:
        try:
            response = requests.get(status_url, headers=headers)
            response.raise_for_status()
            result = response.json()
            status = result.get("status")
            print(f"  📊 {test_name} Job {job_id}: {status}")
            
            if status == "COMPLETED":
                return result.get("output")
            elif status in ["FAILED", "CANCELLED"]:
                print(f"  ❌ {test_name} Failed: {result}")
                return None
            time.sleep(3)
        except Exception as e:
            print(f"  ❌ {test_name} Error: {e}")
            time.sleep(5)

def test_regular_voice_no_userid():
    """Test regular voice without user_id (should work)"""
    print("🎤 Testing Regular Voice: en_man (No user_id)")
    print("=" * 50)
    
    payload = {
        "input": {
            "ref_audio_name": "en_man",
            "text": "Hello, this is a test of the regular voice system without user_id.",
            "seed": 12345,
            "max_new_tokens": 1024
        }
    }
    
    print(f"🎤 Using regular voice: {payload['input']['ref_audio_name']}")
    print(f"📝 Transcript: {payload['input']['text']}")
    print("🔑 No user_id provided (should work for regular voices)")
    
    try:
        response = requests.post(
            RUNPOD_ENDPOINT,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Request successful!")
            
            job_id = result.get("id")
            if not job_id:
                print("❌ No job ID in response")
                return None
            
            print(f"🆔 Job ID: {job_id}")
            
            output = poll_job_status(job_id, "Regular Voice (No user_id)")
            
            if output and "audio_base64" in output:
                print("🎵 Regular voice audio generation completed!")
                
                timestamp = int(time.time())
                filename = f"regular_voice_no_userid_{timestamp}.wav"
                audio_data = base64.b64decode(output["audio_base64"])
                with open(filename, "wb") as f:
                    f.write(audio_data)
                
                print(f"💾 Audio saved: {filename}")
                return filename
            else:
                print("⚠️ No audio in output or generation failed")
                return None
        else:
            print(f"❌ Request failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error calling RunPod endpoint: {str(e)}")
        return None

def test_custom_voice_with_userid():
    """Test custom voice with user_id (should work)"""
    print("\n🎭 Testing Custom Voice: Maya with user_id")
    print("=" * 50)
    
    payload = {
        "input": {
            "user_id": "krxbH4KpoYTWaZKIKIDIzkJx62",
            "ref_audio_name": "cloned_1756205093378_bzbfy4n4x_Maya_Pop_Culture_Queen",
            "text": "Hello, this is a test of the custom voice system with user_id.",
            "seed": 12345,
            "max_new_tokens": 1024
        }
    }
    
    print(f"🔑 User ID: {payload['input']['user_id']}")
    print(f"🎤 Using custom voice: {payload['input']['ref_audio_name']}")
    print(f"📝 Transcript: {payload['input']['text']}")
    
    try:
        response = requests.post(
            RUNPOD_ENDPOINT,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Request successful!")
            
            job_id = result.get("id")
            if not job_id:
                print("❌ No job ID in response")
                return None
            
            print(f"🆔 Job ID: {job_id}")
            
            output = poll_job_status(job_id, "Custom Voice (With user_id)")
            
            if output and "audio_base64" in output:
                print("🎵 Custom voice audio generation completed!")
                
                timestamp = int(time.time())
                filename = f"custom_voice_with_userid_{timestamp}.wav"
                audio_data = base64.b64decode(output["audio_base64"])
                with open(filename, "wb") as f:
                    f.write(audio_data)
                
                print(f"💾 Audio saved: {filename}")
                return filename
            else:
                print("⚠️ No audio in output or generation failed")
                return None
        else:
            print(f"❌ Request failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error calling RunPod endpoint: {str(e)}")
        return None

def test_custom_voice_without_userid():
    """Test custom voice without user_id (should fail)"""
    print("\n❌ Testing Custom Voice: Maya without user_id (Should Fail)")
    print("=" * 60)
    
    payload = {
        "input": {
            "ref_audio_name": "cloned_1756205093378_bzbfy4n4x_Maya_Pop_Culture_Queen",
            "text": "Hello, this should fail because no user_id is provided.",
            "seed": 12345,
            "max_new_tokens": 1024
        }
    }
    
    print(f"🎤 Using custom voice: {payload['input']['ref_audio_name']}")
    print(f"📝 Transcript: {payload['input']['text']}")
    print("❌ No user_id provided (should fail for custom voices)")
    
    try:
        response = requests.post(
            RUNPOD_ENDPOINT,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            print("⚠️ Request succeeded but should have failed!")
            print(f"Response: {result}")
            return "UNEXPECTED_SUCCESS"
        else:
            print(f"✅ Request failed as expected: {response.status_code}")
            print(f"Error: {response.text}")
            return "EXPECTED_FAILURE"
            
    except Exception as e:
        print(f"❌ Error calling RunPod endpoint: {str(e)}")
        return None

def main():
    """Main function to test the new API structure"""
    print("🚀 New API Structure Test")
    print("=" * 80)
    print("This will test:")
    print("1. Regular voice without user_id (should work)")
    print("2. Custom voice with user_id (should work)")
    print("3. Custom voice without user_id (should fail)")
    print("=" * 80)
    
    results = {}
    
    # Test 1: Regular voice without user_id
    print("\n" + "=" * 80)
    print("🎤 PHASE 1: Testing Regular Voice (No user_id)")
    print("=" * 80)
    results["Regular Voice (No user_id)"] = test_regular_voice_no_userid()
    
    # Test 2: Custom voice with user_id
    print("\n" + "=" * 80)
    print("🎭 PHASE 2: Testing Custom Voice (With user_id)")
    print("=" * 80)
    results["Custom Voice (With user_id)"] = test_custom_voice_with_userid()
    
    # Test 3: Custom voice without user_id (should fail)
    print("\n" + "=" * 80)
    print("❌ PHASE 3: Testing Custom Voice (No user_id - Should Fail)")
    print("=" * 80)
    results["Custom Voice (No user_id)"] = test_custom_voice_without_userid()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 NEW API STRUCTURE TEST RESULTS")
    print("=" * 80)
    
    for test_name, result in results.items():
        if result == "EXPECTED_FAILURE":
            status = "✅ PASSED (Failed as expected)"
        elif result == "UNEXPECTED_SUCCESS":
            status = "❌ FAILED (Succeeded when should have failed)"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        
        print(f"  {test_name}: {status}")
        if result and result not in ["EXPECTED_FAILURE", "UNEXPECTED_SUCCESS"]:
            print(f"    📁 File: {result}")
    
    successful = sum(1 for r in results.values() if r and r != "UNEXPECTED_SUCCESS")
    total = len(results)
    print(f"\n🎯 SUCCESS RATE: {successful}/{total}")
    
    if successful == total:
        print("\n🎉 NEW API STRUCTURE WORKING PERFECTLY!")
        print("   - Regular voices work without user_id ✅")
        print("   - Custom voices work with user_id ✅")
        print("   - Custom voices fail without user_id ✅")
        print("   - User isolation working correctly ✅")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
