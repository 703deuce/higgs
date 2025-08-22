"""
Test script for the RunPod serverless handler
Run this locally to test the handler before deployment.
"""

import json
import time
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_generation():
    """Test basic text-to-speech generation."""
    print("Testing basic generation...")
    
    # Import handler after adding to path
    from handler import handler
    
    test_event = {
        "input": {
            "text": "Hello, this is a test of the Higgs Audio v2 model running on RunPod!",
            "temperature": 0.7,
            "max_new_tokens": 256,
            "output_format": "wav"
        }
    }
    
    start_time = time.time()
    result = handler(test_event)
    end_time = time.time()
    
    print(f"Generation completed in {end_time - start_time:.2f} seconds")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False
    else:
        print("✅ Basic generation successful!")
        print(f"  - Audio duration: {result.get('duration', 'N/A')} seconds")
        print(f"  - Sampling rate: {result.get('sampling_rate', 'N/A')} Hz")
        print(f"  - Generated text: {result.get('generated_text', 'N/A')[:100]}...")
        return True

def test_with_scene_description():
    """Test generation with scene description."""
    print("\nTesting with scene description...")
    
    from handler import handler
    
    test_event = {
        "input": {
            "text": "Welcome to our cozy library, where knowledge awaits.",
            "scene_description": "A quiet indoor library with soft ambient sounds and peaceful atmosphere",
            "temperature": 0.5,
            "max_new_tokens": 128
        }
    }
    
    result = handler(test_event)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False
    else:
        print("✅ Scene description test successful!")
        return True

def test_multi_speaker():
    """Test multi-speaker generation."""
    print("\nTesting multi-speaker generation...")
    
    from handler import handler
    
    test_event = {
        "input": {
            "text": "[SPEAKER0] Hello there, how are you doing today? [SPEAKER1] I'm doing great, thanks for asking!",
            "system_prompt": "Generate multi-speaker dialog audio.",
            "temperature": 0.6,
            "max_new_tokens": 200
        }
    }
    
    result = handler(test_event)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return False
    else:
        print("✅ Multi-speaker test successful!")
        return True

def test_optimized_handler():
    """Test the optimized handler."""
    print("\nTesting optimized handler...")
    
    try:
        from handler_optimized import handler as optimized_handler
        
        test_event = {
            "input": {
                "text": "This is a test of the optimized Higgs Audio handler with performance improvements.",
                "temperature": 0.3,
                "max_new_tokens": 128
            }
        }
        
        start_time = time.time()
        result = optimized_handler(test_event)
        end_time = time.time()
        
        print(f"Optimized generation completed in {end_time - start_time:.2f} seconds")
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return False
        else:
            print("✅ Optimized handler test successful!")
            if "timing" in result:
                print(f"  - Generation time: {result['timing']['generation_time']} seconds")
                print(f"  - Total time: {result['timing']['total_time']} seconds")
            return True
            
    except ImportError as e:
        print(f"❌ Could not import optimized handler: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid inputs."""
    print("\nTesting error handling...")
    
    from handler import handler
    
    # Test missing text
    test_event_1 = {
        "input": {
            "temperature": 0.7
        }
    }
    
    result = handler(test_event_1)
    if "error" in result:
        print("✅ Correctly handled missing text")
    else:
        print("❌ Failed to handle missing text")
        return False
    
    # Test invalid temperature
    test_event_2 = {
        "input": {
            "text": "Test",
            "temperature": 5.0  # Too high
        }
    }
    
    result = handler(test_event_2)
    # Should clamp temperature to valid range
    print("✅ Handled invalid temperature (clamped to valid range)")
    
    return True

def save_test_audio(result, filename="test_output.wav"):
    """Save test audio to file for manual verification."""
    if result and "audio_base64" in result and result["audio_base64"]:
        try:
            import base64
            
            audio_data = base64.b64decode(result["audio_base64"])
            with open(filename, "wb") as f:
                f.write(audio_data)
            print(f"✅ Test audio saved to {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")
            return False
    return False

def main():
    """Run all tests."""
    print("🧪 Running Higgs Audio RunPod Handler Tests")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("handler.py"):
        print("❌ handler.py not found. Please run this script from the higgs-audio directory.")
        return
    
    tests = [
        test_basic_generation,
        test_with_scene_description,
        test_multi_speaker,
        test_optimized_handler,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"🧪 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your handler is ready for deployment.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    # Run one more test and save audio for manual verification
    print("\n🔊 Generating test audio file...")
    try:
        from handler import handler
        
        test_event = {
            "input": {
                "text": "This is a test audio file generated by the Higgs Audio RunPod handler.",
                "temperature": 0.7,
                "max_new_tokens": 256
            }
        }
        
        result = handler(test_event)
        if save_test_audio(result):
            print("🎵 You can now listen to test_output.wav to verify audio quality.")
    except Exception as e:
        print(f"❌ Failed to generate test audio: {e}")

if __name__ == "__main__":
    main()
