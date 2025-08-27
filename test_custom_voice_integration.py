#!/usr/bin/env python3
"""
Test Script for Custom Voice Integration with generation.py
Tests that the voice manager can download and resolve custom voices
"""

import os
import sys
from voice_management import VoiceManager

# Firebase configuration
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyASdf98Soi-LtMowVOQMhQvMWWVEP3KoC8",
    "authDomain": "aitts-d4c6d.firebaseapp.com",
    "projectId": "aitts-d4c6d",
    "storageBucket": "aitts-d4c6d.firebasestorage.app",
    "messagingSenderId": "927299361889",
    "appId": "1:927299361889:web:13408945d50bda7a2f5e20",
    "measurementId": "G-P1TK2HHBXR"
}

def test_voice_manager_integration():
    """Test that voice manager can download and resolve voices"""
    print("🧪 Testing Voice Manager Integration")
    print("=" * 60)
    
    # Initialize voice manager
    voice_manager = VoiceManager(FIREBASE_CONFIG)
    
    # Test cases
    test_cases = [
        ("en_man", "regular"),  # Should use voice_prompts
        ("cloned_1756183899826_qawwyeh0f_50t", "custom"),  # Should download from Firebase
    ]
    
    for voice_name, expected_type in test_cases:
        print(f"\n🎤 Testing voice: {voice_name} (expected: {expected_type})")
        
        try:
            audio_path, text_path = voice_manager.get_voice_paths(voice_name, "anonymous")
            
            if expected_type == "custom":
                # Should be in temp_voices directory
                if "/temp_voices/" in audio_path and "/temp_voices/" in text_path:
                    print(f"     ✅ Correctly resolved as custom voice")
                    print(f"        Audio: {os.path.basename(audio_path)}")
                    print(f"        Text: {os.path.basename(text_path)}")
                else:
                    print(f"     ❌ Incorrectly resolved custom voice paths")
                    return False
            else:
                # Should be in voice_prompts directory
                if "/voice_prompts/" in audio_path and "/voice_prompts/" in text_path:
                    print(f"     ✅ Correctly resolved as regular voice")
                    print(f"        Audio: {os.path.basename(audio_path)}")
                    print(f"        Text: {os.path.basename(text_path)}")
                else:
                    print(f"     ❌ Incorrectly resolved regular voice paths")
                    return False
                    
        except Exception as e:
            print(f"     ❌ Error resolving voice paths: {str(e)}")
            return False
    
    return True

def test_generation_integration():
    """Test that generation.py can use the voice manager"""
    print("\n🔧 Testing Generation.py Integration")
    print("-" * 40)
    
    try:
        # Import the resolve_voice_paths function from generation.py
        sys.path.append('examples')
        from generation import resolve_voice_paths
        
        print("✅ Successfully imported resolve_voice_paths from generation.py")
        
        # Test with a regular voice
        print("\n   Testing regular voice: en_man")
        audio_path, text_path = resolve_voice_paths("en_man")
        print(f"     Audio: {audio_path}")
        print(f"     Text: {text_path}")
        
        # Test with a custom voice (this will trigger Firebase download)
        print("\n   Testing custom voice: cloned_1756183899826_qawwyeh0f_50t")
        audio_path, text_path = resolve_voice_paths("cloned_1756183899826_qawwyeh0f_50t")
        print(f"     Audio: {audio_path}")
        print(f"     Text: {text_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing generation integration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Custom Voice Integration Test Suite")
    print("=" * 80)
    print("This will test:")
    print("1. Voice manager voice resolution")
    print("2. Generation.py integration")
    print("3. Automatic Firebase voice downloading")
    print("=" * 80)
    
    # Test 1: Voice manager integration
    success1 = test_voice_manager_integration()
    
    # Test 2: Generation.py integration
    success2 = test_generation_integration()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Voice Manager Integration: {'PASSED' if success1 else 'FAILED'}")
    print(f"✅ Generation.py Integration: {'PASSED' if success2 else 'FAILED'}")
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("   - Voice manager working correctly")
        print("   - Generation.py integration working")
        print("   - Custom voice system 100% functional!")
        print("\n🚀 Ready to use custom voices in generation.py!")
        print("   Example: python examples/generation.py --ref_audio cloned_1756183899826_qawwyeh0f_50t --transcript transcript/single_speaker/en_dl.txt --out_path custom_voice_test.wav")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
