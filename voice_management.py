import os
import tempfile
import base64
import json
import time
import requests
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceManager:
    """
    Manages custom user voices for Higgs Audio V2 SaaS using direct Firebase Storage REST API
    
    Features:
    - Upload custom voices to Firebase Storage (direct REST API)
    - Store voice metadata in simple JSON files (no Firestore needed)
    - Download voices for generation
    - Voice validation and processing
    - No Firebase SDK required - just HTTP requests
    """
    
    def __init__(self, firebase_config: Dict[str, str] = None):
        """Initialize Firebase connection using direct REST API"""
        # Use the same Firebase config as the provided code
        self.firebase_config = firebase_config or {
            "apiKey": "AIzaSyASdf98Soi-LtMowVOQMhQvMWWVEP3KoC8",
            "authDomain": "aitts-d4c6d.firebaseapp.com",
            "projectId": "aitts-d4c6d",
            "storageBucket": "aitts-d4c6d.firebasestorage.app",
            "messagingSenderId": "927299361889",
            "appId": "1:927299361889:web:13408945d50bda7a2f5e20",
            "measurementId": "G-P1TK2HHBXR"
        }
        
        self.storage_bucket = self.firebase_config["storageBucket"]
        self.voices_metadata_file = "/runpod-volume/voices_metadata.json"
        
        # Create metadata file if it doesn't exist
        self._init_metadata_file()
        
        logger.info(f"Voice Manager initialized with Firebase bucket: {self.storage_bucket}")
    
    def _init_metadata_file(self):
        """Initialize the voices metadata JSON file"""
        try:
            if not os.path.exists(self.voices_metadata_file):
                initial_metadata = {
                    "voices": {},
                    "users": {},
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.voices_metadata_file), exist_ok=True)
                
                with open(self.voices_metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(initial_metadata, f, indent=2)
                
                logger.info(f"Created voices metadata file: {self.voices_metadata_file}")
            else:
                logger.info(f"Using existing voices metadata file: {self.voices_metadata_file}")
                
        except Exception as e:
            logger.error(f"Failed to initialize metadata file: {e}")
            raise
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load voices metadata from JSON file"""
        try:
            with open(self.voices_metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return {"voices": {}, "users": {}}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save voices metadata to JSON file"""
        try:
            with open(self.voices_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def upload_to_firebase_storage(self, audio_bytes: bytes, filename: str) -> str:
        """
        Upload audio file to Firebase Storage using direct REST API
        Same approach as the provided code
        """
        try:
            logger.info(f"🔼 Uploading {len(audio_bytes)} bytes to Firebase Storage: {filename}")
            
            # Firebase Storage REST API endpoint
            upload_url = f"https://firebasestorage.googleapis.com/v0/b/{self.storage_bucket}/o"
            
            # Upload file using multipart form data
            files = {
                'file': (filename, audio_bytes, 'audio/wav')
            }
            
            params = {
                'name': filename,
                'uploadType': 'multipart'
            }
            
            # Make upload request
            response = requests.post(
                upload_url,
                params=params,
                files=files,
                timeout=120  # 2 minute timeout for upload
            )
            
            if response.status_code == 200:
                upload_result = response.json()
                
                # Get download URL
                download_url = f"https://firebasestorage.googleapis.com/v0/b/{self.storage_bucket}/o/{filename.replace('/', '%2F')}?alt=media"
                
                file_size_mb = len(audio_bytes) / 1024 / 1024
                logger.info(f"✅ Firebase upload successful: {file_size_mb:.1f}MB uploaded")
                logger.info(f"📥 Download URL: {download_url[:50]}...")
                
                return download_url
                
            else:
                error_msg = f"Firebase upload failed: {response.status_code} - {response.text}"
                logger.error(f"❌ {error_msg}")
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"❌ Firebase upload error: {str(e)}")
            raise Exception(f"Firebase upload failed: {str(e)}")
    
    def download_from_firebase(self, firebase_url: str) -> str:
        """
        Download audio file from Firebase Storage URL
        Same approach as the provided code
        """
        try:
            logger.info(f"🔽 Downloading audio from Firebase: {firebase_url}")
            
            # Download the file
            response = requests.get(firebase_url, stream=True)
            response.raise_for_status()
            
            # Create temporary file in Firebase temp voices directory
            firebase_temp_dir = "/runpod-volume/temp_voices"
            os.makedirs(firebase_temp_dir, exist_ok=True)
            
            # Generate unique filename
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4())[:8]
            temp_filename = f"temp_firebase_{timestamp}_{unique_id}.wav"
            temp_file_path = os.path.join(firebase_temp_dir, temp_filename)
            
            # Download with progress logging
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192
            
            with open(temp_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (chunk_size * 100) == 0:  # Log every 100 chunks
                                logger.info(f"📥 Download progress: {progress:.1f}% ({downloaded}/{total_size} bytes)")
            
            # Verify file was downloaded
            file_size = os.path.getsize(temp_file_path)
            logger.info(f"✅ Firebase download complete: {temp_file_path} ({file_size} bytes, {file_size/1024/1024:.1f} MB)")
            
            return temp_file_path
            
        except Exception as e:
            logger.error(f"❌ Failed to download from Firebase: {str(e)}")
            raise Exception(f"Firebase download failed: {str(e)}")
    
    def upload_custom_voice(
        self, 
        user_id: str, 
        voice_name: str, 
        audio_base64: str, 
        transcription: str,
        voice_description: str = None
    ) -> Dict[str, Any]:
        """
        Upload a custom voice for a user using direct Firebase Storage
        """
        try:
            # Validate audio data
            audio_data = base64.b64decode(audio_base64)
            if len(audio_data) < 1000:  # Basic size validation
                raise ValueError("Audio file too small")
            
            # Create unique voice ID
            voice_id = f"{user_id}_{voice_name}_{int(time.time())}"
            
            # Generate Firebase Storage filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            firebase_filename = f"voices/{user_id}/{voice_id}.wav"
            
            # Upload to Firebase Storage
            firebase_url = self.upload_to_firebase_storage(audio_data, firebase_filename)
            
            # Load current metadata
            metadata = self._load_metadata()
            
            # Create voice record
            voice_record = {
                'voice_id': voice_id,
                'user_id': user_id,
                'voice_name': voice_name,
                'firebase_url': firebase_url,
                'firebase_filename': firebase_filename,
                'transcription': transcription,
                'description': voice_description,
                'created_at': datetime.now().isoformat(),
                'status': 'active',
                'usage_count': 0,
                'file_size_bytes': len(audio_data)
            }
            
            # Save to metadata
            metadata['voices'][voice_id] = voice_record
            
            # Update user's voice list
            if user_id not in metadata['users']:
                metadata['users'][user_id] = []
            metadata['users'][user_id].append(voice_id)
            
            # Save metadata
            self._save_metadata(metadata)
            
            logger.info(f"Voice uploaded successfully: {voice_id}")
            
            return {
                'success': True,
                'voice_id': voice_id,
                'voice_name': voice_name,
                'firebase_url': firebase_url,
                'message': 'Voice uploaded successfully'
            }
            
        except Exception as e:
            logger.error(f"Failed to upload voice: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to upload voice'
            }
    
    def get_user_voices(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all voices for a specific user"""
        try:
            metadata = self._load_metadata()
            user_voices = []
            
            if user_id in metadata['users']:
                for voice_id in metadata['users'][user_id]:
                    if voice_id in metadata['voices']:
                        voice_data = metadata['voices'][voice_id].copy()
                        voice_data['id'] = voice_id
                        user_voices.append(voice_data)
            
            return user_voices
            
        except Exception as e:
            logger.error(f"Failed to get user voices: {e}")
            return []
    
    def download_voice_for_generation(self, voice_id: str) -> Optional[Dict[str, Any]]:
        """
        Download a voice for use in generation.py
        """
        try:
            # Load metadata
            metadata = self._load_metadata()
            
            if voice_id not in metadata['voices']:
                logger.error(f"Voice {voice_id} not found")
                return None
            
            voice_data = metadata['voices'][voice_id]
            
            # Download from Firebase
            local_audio_path = self.download_from_firebase(voice_data['firebase_url'])
            
            # Create transcription file
            firebase_temp_dir = "/runpod-volume/temp_voices"
            temp_voice_name = os.path.splitext(os.path.basename(local_audio_path))[0]
            text_path = os.path.join(firebase_temp_dir, f"{temp_voice_name}.txt")
            
            # Save transcription
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(voice_data['transcription'])
            
            # Update usage count
            metadata['voices'][voice_id]['usage_count'] += 1
            self._save_metadata(metadata)
            
            logger.info(f"Voice downloaded for generation: {temp_voice_name}")
            
            return {
                'voice_name': temp_voice_name,
                'audio_path': local_audio_path,
                'text_path': text_path,
                'transcription': voice_data['transcription']
            }
            
        except Exception as e:
            logger.error(f"Failed to download voice {voice_id}: {e}")
            return None
    
    def cleanup_temp_voice(self, temp_voice_name: str):
        """Clean up temporary voice files after generation"""
        try:
            firebase_temp_dir = "/runpod-volume/temp_voices"
            
            for ext in ['.wav', '.txt']:
                file_path = os.path.join(firebase_temp_dir, f"{temp_voice_name}{ext}")
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Cleaned up: {file_path}")
                    
        except Exception as e:
            logger.warning(f"Failed to cleanup temp voice {temp_voice_name}: {e}")
    
    def delete_voice(self, voice_id: str, user_id: str) -> bool:
        """Delete a voice (soft delete by setting status to inactive)"""
        try:
            metadata = self._load_metadata()
            
            if voice_id not in metadata['voices']:
                return False
            
            voice_data = metadata['voices'][voice_id]
            if voice_data['user_id'] != user_id:
                logger.warning(f"User {user_id} attempted to delete voice {voice_id} owned by {voice_data['user_id']}")
                return False
            
            # Soft delete
            metadata['voices'][voice_id]['status'] = 'deleted'
            
            # Remove from user's voice list
            if user_id in metadata['users'] and voice_id in metadata['users'][user_id]:
                metadata['users'][user_id].remove(voice_id)
            
            # Save metadata
            self._save_metadata(metadata)
            
            logger.info(f"Voice {voice_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete voice {voice_id}: {e}")
            return False
    
    def validate_voice_upload(self, audio_base64: str, transcription: str) -> Dict[str, Any]:
        """Validate voice upload requirements"""
        try:
            # Decode audio
            audio_data = base64.b64decode(audio_base64)
            
            # Check file size (should be reasonable for 10-second audio)
            if len(audio_data) < 10000:  # 10KB minimum
                return {'valid': False, 'error': 'Audio file too small (minimum 10KB)'}
            
            if len(audio_data) > 5000000:  # 5MB maximum
                return {'valid': False, 'error': 'Audio file too large (maximum 5MB)'}
            
            # Check transcription length
            if len(transcription.strip()) < 10:
                return {'valid': False, 'error': 'Transcription too short (minimum 10 characters)'}
            
            if len(transcription.strip()) > 500:
                return {'valid': False, 'error': 'Transcription too long (maximum 500 characters)'}
            
            return {'valid': True, 'message': 'Voice upload validation passed'}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}


# Example usage and testing
if __name__ == "__main__":
    # Initialize voice manager with the same Firebase config
    firebase_config = {
        "apiKey": "AIzaSyASdf98Soi-LtMowVOQMhQvMWWVEP3KoC8",
        "authDomain": "aitts-d4c6d.firebaseapp.com",
        "projectId": "aitts-d4c6d",
        "storageBucket": "aitts-d4c6d.firebasestorage.app",
        "messagingSenderId": "927299361889",
        "appId": "1:927299361889:web:13408945d50bda7a2f5e20",
        "measurementId": "G-P1TK2HHBXR"
    }
    
    voice_manager = VoiceManager(firebase_config)
    
    # Example: Upload a custom voice
    test_audio_base64 = "UklGRn..."  # Your base64 audio here
    test_transcription = "Hello, this is my custom voice for podcasting."
    
    result = voice_manager.upload_custom_voice(
        user_id="user123",
        voice_name="my_podcast_voice",
        audio_base64=test_audio_base64,
        transcription=test_transcription,
        voice_description="Professional podcast voice with clear articulation"
    )
    
    print(f"Upload result: {result}")
