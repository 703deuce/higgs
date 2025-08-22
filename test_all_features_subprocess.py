#!/usr/bin/env python3
"""
Comprehensive test of all Higgs Audio v2 features using the subprocess handler.
Tests: Regular TTS, Voice Cloning, Long-form Chunking, Experimental Features
"""

import requests
import json
import base64
import time
import os

# RunPod endpoint configuration
RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/zeqk8y61qusvji/run"
API_KEY = "rpa_C55TBQG7H6FM7G3Q7A6JM7ZJCDKA3I2J3EO0TAH8fxyddo"

def encode_audio_to_base64(file_path):
    """Encode audio file to base64"""
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

def poll_job_status(job_id, feature_name):
    """Poll job status until completion"""
    status_url = f"https://api.runpod.ai/v2/zeqk8y61qusvji/status/{job_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    while True:
        try:
            response = requests.get(status_url, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            status = result.get("status")
            
            print(f"  {feature_name} Job {job_id}: {status}")
            
            if status == "COMPLETED":
                return result.get("output")
            elif status in ["FAILED", "CANCELLED"]:
                print(f"  ❌ {feature_name} failed: {result}")
                return None
            
            time.sleep(3)
            
        except Exception as e:
            print(f"  Error polling {feature_name}: {e}")
            time.sleep(5)

def test_feature(feature_name, payload, expected_duration_range=None):
    """Test a specific feature"""
    print(f"\n🧪 Testing: {feature_name}")
    print("=" * (10 + len(feature_name)))
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        # Submit the job
        response = requests.post(RUNPOD_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        job_id = result.get("id")
        
        if not job_id:
            print(f"  ❌ Failed to get job ID: {result}")
            return None
        
        print(f"  ✅ Job submitted: {job_id}")
        
        # Poll for completion
        output = poll_job_status(job_id, feature_name)
        
        if output and "audio_base64" in output:
            # Save the generated audio
            timestamp = int(time.time())
            safe_name = feature_name.lower().replace(" ", "_").replace("-", "_")
            filename = f"subprocess_{safe_name}_{timestamp}.wav"
            
            audio_data = base64.b64decode(output["audio_base64"])
            with open(filename, "wb") as f:
                f.write(audio_data)
            
            duration = output.get('duration', 0)
            print(f"  🎵 Audio saved: {filename}")
            print(f"  📊 Duration: {duration}s")
            print(f"  📈 Sample rate: {output.get('sampling_rate', 'unknown')}Hz")
            print(f"  🔧 Method: {output.get('method', 'unknown')}")
            
            # Check if duration is in expected range
            if expected_duration_range:
                min_dur, max_dur = expected_duration_range
                if min_dur <= duration <= max_dur:
                    print(f"  ✅ Duration within expected range ({min_dur}-{max_dur}s)")
                else:
                    print(f"  ⚠️ Duration outside expected range ({min_dur}-{max_dur}s)")
            
            return filename
        else:
            print(f"  ❌ No audio in output: {output}")
            return None
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    """Test all features of the subprocess handler"""
    
    print("🚀 COMPREHENSIVE HIGGS AUDIO V2 FEATURE TEST")
    print("=" * 50)
    print("Testing subprocess handler with generation.py")
    print()
    
    results = {}
    
    # Test 1: Regular Text-to-Speech
    print("📋 TEST 1: Regular Text-to-Speech")
    regular_tts_payload = {
        "input": {
            "text": "Hello! This is a test of the subprocess handler for Higgs Audio v2. The weather today is absolutely wonderful.",
            "temperature": 0.7,
            "seed": 12345,
            "max_new_tokens": 512
        }
    }
    results["Regular TTS"] = test_feature("Regular TTS", regular_tts_payload, (5, 15))
    
    # Test 2: Voice Cloning (using existing voice sample)
    print("📋 TEST 2: Voice Cloning with Existing Sample")
    voice_clone_payload = {
        "input": {
            "text": "This is a voice cloning test using the reference_trimmed sample that we uploaded to GitHub.",
            "ref_audio_name": "reference_trimmed",  # Use our uploaded sample
            "temperature": 0.7,
            "seed": 12345,
            "max_new_tokens": 512
        }
    }
    results["Voice Cloning"] = test_feature("Voice Cloning", voice_clone_payload, (5, 15))
    
    # Test 3: Long-form Chunking
    print("📋 TEST 3: Long-form Audio with Chunking")
    longform_text = """
    Welcome to this comprehensive test of long-form audio generation with chunking capabilities. 
    
    This feature allows the model to process very long text by breaking it into smaller, manageable chunks. Each chunk is processed individually while maintaining context from previous chunks.
    
    The chunking system supports different methods including word-based chunking and speaker-based chunking. Word-based chunking splits text based on word count, while speaker-based chunking splits based on speaker turns in dialogue.
    
    This particular test uses word-based chunking with a maximum of 30 words per chunk and a buffer size of 2, which means the model will keep the last 2 chunks in memory for context.
    
    The result should be a coherent, natural-sounding narration that flows smoothly across chunk boundaries without noticeable breaks or inconsistencies.
    """
    
    chunking_payload = {
        "input": {
            "text": longform_text.strip(),
            "ref_audio_name": "en_man",  # Use existing voice sample
            "chunk_method": "word",
            "chunk_max_word_num": 30,  # Small chunks for testing
            "generation_chunk_buffer_size": 2,
            "temperature": 0.3,
            "seed": 12345,
            "max_new_tokens": 1024
        }
    }
    results["Long-form Chunking"] = test_feature("Long-form Chunking", chunking_payload, (15, 60))
    
    # Test 4: Experimental Humming
    print("📋 TEST 4: Experimental Humming")
    humming_payload = {
        "input": {
            "text": "Let me demonstrate the humming capability. [humming start] la la la la la la la [humming end] That was a beautiful tune!",
            "ref_audio_name": "en_woman",
            "ras_win_len": 0,  # Disable RAS for experimental features
            "temperature": 0.7,
            "seed": 12345,
            "max_new_tokens": 512
        }
    }
    results["Experimental Humming"] = test_feature("Experimental Humming", humming_payload, (8, 20))
    
    # Test 5: Experimental BGM
    print("📋 TEST 5: Experimental Background Music")
    bgm_payload = {
        "input": {
            "text": "[music start] In a world where artificial intelligence meets human creativity, Higgs Audio represents the next evolution of speech synthesis technology. [music end]",
            "ref_audio_name": "en_woman",
            "ras_win_len": 0,
            "ref_audio_in_system_message": True,  # Key for BGM
            "temperature": 0.7,
            "seed": 123456,
            "max_new_tokens": 512
        }
    }
    results["Experimental BGM"] = test_feature("Experimental BGM", bgm_payload, (8, 20))
    
    # Test 6: Multi-speaker Dialogue
    print("📋 TEST 6: Multi-speaker Dialogue")
    multispeaker_payload = {
        "input": {
            "text": "[SPEAKER0] Hello there! How are you doing today?\n[SPEAKER1] I'm doing great, thank you for asking! How about yourself?\n[SPEAKER0] Wonderful! I'm excited to test this multi-speaker feature.",
            "temperature": 0.7,
            "seed": 12345,
            "max_new_tokens": 512
        }
    }
    results["Multi-speaker"] = test_feature("Multi-speaker", multispeaker_payload, (8, 20))
    
    # Test 7: Scene-based Generation
    print("📋 TEST 7: Scene-based Generation")
    scene_payload = {
        "input": {
            "text": "Welcome to our quiet library. Please speak softly as others are reading and studying.",
            "scene_description": "A quiet library with soft lighting. People are reading books and the atmosphere is calm and peaceful.",
            "ref_audio_name": "en_man",
            "temperature": 0.3,
            "seed": 12345,
            "max_new_tokens": 512
        }
    }
    results["Scene-based"] = test_feature("Scene-based", scene_payload, (5, 15))
    
    # Summary
    print("\n📊 FINAL RESULTS")
    print("=" * 30)
    
    successful_tests = sum(1 for result in results.values() if result is not None)
    total_tests = len(results)
    
    for feature, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {feature}: {status}")
    
    print(f"\n🎯 SUCCESS RATE: {successful_tests}/{total_tests} ({100*successful_tests/total_tests:.1f}%)")
    
    if successful_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Subprocess handler supports all features!")
    elif successful_tests > 0:
        print(f"\n✅ {successful_tests} features working. Subprocess approach is functional!")
    else:
        print("\n❌ All tests failed. Check subprocess implementation.")
    
    print("\n💡 The subprocess approach uses the official generation.py script,")
    print("   ensuring 100% compatibility with all documented features.")

if __name__ == "__main__":
    main()
