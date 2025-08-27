#!/usr/bin/env python3
"""
Test Script for Firebase Voice Download
Tests downloading custom voices from Firebase Storage
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

def test_custom_voice_download():
    """Test downloading a custom voice from Firebase"""
    print("🧪 Testing Custom Voice Download from Firebase")
    print("=" * 60)
    
    # Initialize voice manager
    print("🔧 Initializing Voice Manager...")
    voice_manager = VoiceManager(FIREBASE_CONFIG)
    
    # Test voice details
    user_id = "anonymous"
    voice_name = "cloned_1756183899826_qawwyeh0f_50t"
    
    print(f"📁 Testing voice: {voice_name}")
    print(f"👤 User ID: {user_id}")
    
    # Test 1: Check if it's a custom voice
    print("\n🔍 Test 1: Check if voice is custom")
    print("-" * 40)
    
    is_custom = voice_manager.is_custom_voice(voice_name)
    print(f"   Is custom voice: {'✅ Yes' if is_custom else '❌ No'}")
    
    if not is_custom:
        print("❌ Voice should be custom but was detected as regular")
        return False
    
    # Test 2: Download custom voice
    print("\n📥 Test 2: Download Custom Voice from Firebase")
    print("-" * 40)
    
    try:
        voice_info = voice_manager.download_custom_voice(user_id, voice_name)
        
        if voice_info:
            print(f"✅ Download successful!")
            print(f"   Audio path: {voice_info['audio_path']}")
            print(f"   Text path: {voice_info['text_path']}")
            print(f"   Voice name: {voice_info['voice_name']}")
            
            # Verify files exist
            audio_exists = os.path.exists(voice_info['audio_path'])
            text_exists = os.path.exists(voice_info['text_path'])
            
            print(f"   Audio file exists: {'✅ Yes' if audio_exists else '❌ No'}")
            print(f"   Text file exists: {'✅ Yes' if text_exists else '❌ No'}")
            
            if audio_exists and text_exists:
                # Get file sizes
                audio_size = os.path.getsize(voice_info['audio_path'])
                text_size = os.path.getsize(voice_info['text_path'])
                
                print(f"   Audio size: {audio_size:,} bytes")
                print(f"   Text size: {text_size:,} bytes")
                
                # Read text content
                with open(voice_info['text_path'], 'r', encoding='utf-8') as f:
                    text_content = f.read()
                print(f"   Text content: {text_content[:100]}...")
                
                return True
            else:
                print("❌ Downloaded files not found")
                return False
        else:
            print("❌ Download failed - no voice info returned")
            return False
            
    except Exception as e:
        print(f"❌ Download test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_voice_path_resolution():
    """Test the voice path resolution system"""
    print("\n🎤 Test 3: Voice Path Resolution")
    print("-" * 40)
    
    # Initialize voice manager
    voice_manager = VoiceManager(FIREBASE_CONFIG)
    
    # Test cases
    test_cases = [
        ("en_man", "regular"),  # Should use voice_prompts
        ("cloned_1756183899826_qawwyeh0f_50t", "custom"),  # Should download from Firebase
    ]
    
    for voice_name, expected_type in test_cases:
        print(f"\n   Testing voice: {voice_name} (expected: {expected_type})")
        
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

def main():
    """Run all tests"""
    print("🚀 Firebase Voice Download Test Suite")
    print("=" * 80)
    print("This will test:")
    print("1. Custom voice detection")
    print("2. Firebase voice download")
    print("3. Voice path resolution system")
    print("=" * 80)
    
    # Test 1: Custom voice download
    success1 = test_custom_voice_download()
    
    # Test 2: Voice path resolution
    success2 = test_voice_path_resolution()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Custom Voice Download: {'PASSED' if success1 else 'FAILED'}")
    print(f"✅ Voice Path Resolution: {'PASSED' if success2 else 'FAILED'}")
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("   - Custom voice download from Firebase working")
        print("   - Voice path resolution system working")
        print("   - Ready to integrate with generation.py!")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
