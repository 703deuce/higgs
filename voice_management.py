#!/usr/bin/env python3
"""
Voice Management System for Higgs Audio
Handles custom voice downloads from Firebase Storage
"""

import os
import json
import tempfile
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple

class VoiceManager:
    def __init__(self, firebase_config: Dict[str, str]):
        """Initialize Voice Manager with Firebase configuration"""
        self.firebase_config = firebase_config
        self.storage_bucket = firebase_config["storageBucket"]
        
        # Create temp voices directory for downloaded custom voices
        self.temp_voices_dir = "/runpod-volume/temp_voices"
        os.makedirs(self.temp_voices_dir, exist_ok=True)
        
        print(f"Voice Manager initialized with Firebase bucket: {self.storage_bucket}")
    
    def download_custom_voice(self, user_id: str, voice_name: str) -> Optional[Dict[str, str]]:
        """
        Download a custom voice from Firebase Storage
        
        Args:
            user_id: User ID (e.g., 'krxbH4KpoYTWaZKIKIDIzkJx62')
            voice_name: Voice name (e.g., 'cloned_1756205093378_bzbfy4n4x_Maya_Pop_Culture_Queen')
        
        Returns:
            Dict with 'audio_path' and 'text_path' if successful, None if failed
        """
        try:
            # Firebase Storage paths - use user-specific folder structure
            audio_path = f"user_voices/{user_id}/{voice_name}.wav"
            text_path = f"user_voices/{user_id}/{voice_name}.txt"
            
            # Local temp paths
            local_audio_path = os.path.join(self.temp_voices_dir, f"{voice_name}.wav")
            local_text_path = os.path.join(self.temp_voices_dir, f"{voice_name}.txt")
            
            print(f"🔽 Downloading custom voice: {voice_name}")
            print(f"   User ID: {user_id}")
            print(f"   Audio: {audio_path}")
            print(f"   Text: {text_path}")
            
            # Download audio file
            audio_url = self._get_download_url(audio_path)
            if not audio_url:
                print(f"❌ Failed to get download URL for audio: {audio_path}")
                return None
            
            audio_success = self._download_file(audio_url, local_audio_path)
            if not audio_success:
                print(f"❌ Failed to download audio file")
                return None
            
            # Download text file
            text_url = self._get_download_url(text_path)
            if not text_url:
                print(f"❌ Failed to get download URL for text: {text_path}")
                return None
            
            text_success = self._download_file(text_url, local_text_path)
            if not text_success:
                print(f"❌ Failed to download text file")
                return None
            
            # Verify files exist and have content
            if not os.path.exists(local_audio_path) or not os.path.exists(local_text_path):
                print(f"❌ Downloaded files not found")
                return None
            
            audio_size = os.path.getsize(local_audio_path)
            text_size = os.path.getsize(local_text_path)
            
            print(f"✅ Custom voice downloaded successfully!")
            print(f"   User ID: {user_id}")
            print(f"   Audio: {local_audio_path} ({audio_size:,} bytes)")
            print(f"   Text: {local_text_path} ({text_size:,} bytes)")
            
            return {
                'audio_path': local_audio_path,
                'text_path': local_text_path,
                'voice_name': voice_name
            }
            
        except Exception as e:
            print(f"❌ Error downloading custom voice: {str(e)}")
            return None
    
    def _get_download_url(self, firebase_path: str) -> Optional[str]:
        """Get download URL for Firebase Storage file"""
        try:
            # URL encode the path for Firebase Storage REST API
            import urllib.parse
            encoded_path = urllib.parse.quote(firebase_path, safe='')
            download_url = f"https://firebasestorage.googleapis.com/v0/b/{self.storage_bucket}/o/{encoded_path}?alt=media"
            return download_url
        except Exception as e:
            print(f"❌ Error getting download URL: {str(e)}")
            return None
    
    def _download_file(self, url: str, local_path: str) -> bool:
        """Download file from URL to local path"""
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                print(f"❌ Download failed with status {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error downloading file: {str(e)}")
            return False
    
    def is_custom_voice(self, voice_name: str) -> bool:
        """Check if a voice name is a custom voice (not in voice_prompts)"""
        voice_prompts_dir = "/app/examples/voice_prompts"
        regular_voice_path = os.path.join(voice_prompts_dir, f"{voice_name}.wav")
        
        # If it's not in voice_prompts, it's a custom voice
        return not os.path.exists(regular_voice_path)
    
    def get_voice_paths(self, voice_name: str, user_id: str = None) -> Tuple[str, str]:
        """
        Get voice audio and text paths, downloading from Firebase if needed
        
        Args:
            voice_name: Name of the voice
            user_id: User ID for custom voices (required if voice_name starts with 'cloned_')
        
        Returns:
            Tuple of (audio_path, text_path)
        """
        if self.is_custom_voice(voice_name):
            if not user_id:
                raise ValueError(f"user_id is required for custom voice: {voice_name}")
            
            print(f"🎤 Using custom voice: {voice_name} for user: {user_id}")
            # Download from Firebase using user-specific folder
            voice_info = self.download_custom_voice(user_id, voice_name)
            if voice_info:
                return voice_info['audio_path'], voice_info['text_path']
            else:
                raise ValueError(f"Failed to download custom voice: {voice_name}")
        else:
            print(f"🎤 Using regular voice: {voice_name}")
            # Use existing voice_prompts
            voice_prompts_dir = "/app/examples/voice_prompts"
            audio_path = os.path.join(voice_prompts_dir, f"{voice_name}.wav")
            text_path = os.path.join(voice_prompts_dir, f"{voice_name}.txt")
            
            if not os.path.exists(audio_path):
                raise ValueError(f"Regular voice audio not found: {audio_path}")
            if not os.path.exists(text_path):
                raise ValueError(f"Regular voice text not found: {text_path}")
            
            return audio_path, text_path
