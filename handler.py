import os
import json
import base64
import tempfile
import subprocess
import time
import logging
from typing import Dict, Any, Optional
from voice_management import VoiceManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Voice Manager with the same Firebase config as the provided code
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

def validate_input(event: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize input parameters with custom voice support"""
    if "input" not in event:
        raise ValueError("Missing 'input' field in request")
    
    input_data = event["input"]
    
    if "text" not in input_data:
        raise ValueError("Missing 'text' field in input")
    
    # Check if user_id is required (only for custom voices)
    ref_audio_name = input_data.get("ref_audio_name")
    if ref_audio_name and ref_audio_name.startswith("cloned_"):
        if "user_id" not in input_data:
            raise ValueError("user_id is required for custom voices (cloned_*)")
    
    # Map API parameters to generation.py arguments
    validated_input = {
        "text": input_data["text"],
        "max_new_tokens": input_data.get("max_new_tokens", 2048),
        "temperature": input_data.get("temperature", 1.0),
        "top_k": input_data.get("top_k", 50),
        "top_p": input_data.get("top_p", 0.95),
        "seed": input_data.get("seed", None),
        "ras_win_len": input_data.get("ras_win_len", 7),
        "ras_win_max_num_repeat": input_data.get("ras_win_max_num_repeat", 2),
        "output_format": input_data.get("output_format", "wav"),
        
        # Reference audio for voice cloning
        "ref_audio_base64": input_data.get("ref_audio_base64", None),
        "ref_audio_text": input_data.get("ref_audio_text", None),
        "ref_audio_name": ref_audio_name,
        
        # Custom voice support
        "custom_voice_id": input_data.get("custom_voice_id", None),
        "user_id": input_data.get("user_id", None),
        
        # Scene and experimental features
        "scene_description": input_data.get("scene_description", None),
        "ref_audio_in_system_message": input_data.get("ref_audio_in_system_message", False),
        
        # Long-form generation chunking parameters
        "chunk_method": input_data.get("chunk_method", None),
        "chunk_max_word_num": input_data.get("chunk_max_word_num", 200),
        "chunk_max_num_turns": input_data.get("chunk_max_num_turns", 1),
        "generation_chunk_buffer_size": input_data.get("generation_chunk_buffer_size", None),
    }
    
    return validated_input

def create_temp_files(validated_input: Dict[str, Any]) -> Dict[str, str]:
    """Create temporary files needed for generation.py with custom voice support"""
    temp_files = {}
    
    # Create transcript file
    transcript_fd, transcript_path = tempfile.mkstemp(suffix='.txt', prefix='transcript_', dir='/tmp')
    with os.fdopen(transcript_fd, 'w', encoding='utf-8') as f:
        f.write(validated_input["text"])
    temp_files['transcript'] = transcript_path
    
    # Create scene prompt file if provided
    if validated_input["scene_description"]:
        scene_fd, scene_path = tempfile.mkstemp(suffix='.txt', prefix='scene_', dir='/tmp')
        with os.fdopen(scene_fd, 'w', encoding='utf-8') as f:
            f.write(validated_input["scene_description"])
        temp_files['scene'] = scene_path
    
    # Handle reference audio with custom voice support
    ref_audio_name = None
    
    # Priority 1: Custom voice from Firebase (requires user_id)
    if validated_input["ref_audio_name"] and validated_input["ref_audio_name"].startswith("cloned_"):
        if not validated_input["user_id"]:
            raise ValueError("user_id is required for custom voices (cloned_*)")
        
        try:
            # Use voice manager to download custom voice
            voice_info = voice_manager.download_custom_voice(
                validated_input["user_id"],
                validated_input["ref_audio_name"]
            )
            if voice_info:
                ref_audio_name = voice_info['voice_name']
                temp_files['custom_voice'] = voice_info
                temp_files['needs_cleanup'] = True
                logger.info(f"Using custom voice: {ref_audio_name} for user: {validated_input['user_id']}")
            else:
                logger.warning(f"Failed to download custom voice: {validated_input['ref_audio_name']}")
                raise ValueError(f"Failed to download custom voice: {validated_input['ref_audio_name']}")
        except Exception as e:
            logger.error(f"Error with custom voice: {e}")
            raise
    
    # Priority 2: Use existing voice sample by name (regular voices)
    elif validated_input["ref_audio_name"]:
        ref_audio_name = validated_input["ref_audio_name"]
        logger.info(f"Using regular voice sample: {ref_audio_name}")
    
    # Priority 3: Create temporary voice sample from base64
    elif validated_input["ref_audio_base64"]:
        # Decode base64 audio
        audio_data = base64.b64decode(validated_input["ref_audio_base64"])
        
        # Create a unique name for this reference audio
        ref_audio_name = f"temp_ref_{int(time.time())}"
        
        # Save reference audio in Firebase temp voices directory
        firebase_temp_dir = "/runpod-volume/temp_voices"
        os.makedirs(firebase_temp_dir, exist_ok=True)
        
        ref_audio_path = os.path.join(firebase_temp_dir, f"{ref_audio_name}.wav")
        with open(ref_audio_path, 'wb') as f:
            f.write(audio_data)
        temp_files['ref_audio'] = ref_audio_path
        
        # Save reference transcription in Firebase temp voices directory
        if validated_input["ref_audio_text"]:
            ref_text_path = os.path.join(firebase_temp_dir, f"{ref_audio_name}.txt")
            with open(ref_text_path, 'w', encoding='utf-8') as f:
                f.write(validated_input["ref_audio_text"])
            temp_files['ref_text'] = ref_text_path
        
        logger.info(f"Created temporary voice sample: {ref_audio_name}")
    
    if ref_audio_name:
        temp_files['ref_name'] = ref_audio_name
    
    # Create output file path
    output_fd, output_path = tempfile.mkstemp(suffix='.wav', prefix='output_', dir='/tmp')
    os.close(output_fd)  # Close the file descriptor, we just need the path
    temp_files['output'] = output_path
    
    return temp_files

def build_generation_command(validated_input: Dict[str, Any], temp_files: Dict[str, str]) -> list:
    """Build the command to run generation.py"""
    cmd = [
        "python3", "/app/examples/generation.py",
        "--transcript", temp_files['transcript'],
        "--max_new_tokens", str(validated_input["max_new_tokens"]),
        "--temperature", str(validated_input["temperature"]),
        "--top_k", str(validated_input["top_k"]),
        "--top_p", str(validated_input["top_p"]),
        "--ras_win_len", str(validated_input["ras_win_len"]),
        "--ras_win_max_num_repeat", str(validated_input["ras_win_max_num_repeat"]),
        "--out_path", temp_files['output']
    ]
    
    # Add seed if provided
    if validated_input["seed"] is not None:
        cmd.extend(["--seed", str(validated_input["seed"])])
    
    # Add user_id if provided (for custom voices)
    if validated_input.get("user_id"):
        cmd.extend(["--user_id", validated_input["user_id"]])
        logger.info(f"🔍 Debug: Added --user_id {validated_input['user_id']} to command")
    
    # Add scene prompt if provided
    if validated_input["scene_description"]:
        cmd.extend(["--scene_prompt", temp_files['scene']])
    
    # Add reference audio if provided
    if 'ref_name' in temp_files:
        cmd.extend(["--ref_audio", temp_files['ref_name']])
        
        # Add experimental flag if needed
        if validated_input["ref_audio_in_system_message"]:
            cmd.append("--ref_audio_in_system_message")
    
    # Add chunking parameters if provided
    if validated_input["chunk_method"]:
        cmd.extend(["--chunk_method", validated_input["chunk_method"]])
        cmd.extend(["--chunk_max_word_num", str(validated_input["chunk_max_word_num"])])
        cmd.extend(["--chunk_max_num_turns", str(validated_input["chunk_max_num_turns"])])
        
        if validated_input["generation_chunk_buffer_size"] is not None:
            cmd.extend(["--generation_chunk_buffer_size", str(validated_input["generation_chunk_buffer_size"])])
    
    return cmd

def run_generation(cmd: list, user_id: str = None) -> tuple:
    """Run the generation.py command and return success status and output"""
    try:
        # Set environment variables for caching
        env = os.environ.copy()
        env.update({
            "HF_HOME": "/runpod-volume/.huggingface",
            "TRANSFORMERS_CACHE": "/runpod-volume/.huggingface/transformers",
            "HF_DATASETS_CACHE": "/runpod-volume/.huggingface/datasets",
            "TORCH_HOME": "/runpod-volume/.torch"
        })
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("Generation completed successfully")
            return True, result.stdout
        else:
            logger.error(f"Generation failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            return False, f"STDERR: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        logger.error("Generation timed out after 10 minutes")
        return False, "Generation timed out"
    except Exception as e:
        logger.error(f"Error running generation: {e}")
        return False, str(e)

def cleanup_temp_files(temp_files: Dict[str, str]):
    """Clean up temporary files including custom voices"""
    try:
        # Clean up custom voice files
        if temp_files.get('needs_cleanup') and 'custom_voice' in temp_files:
            custom_voice = temp_files['custom_voice']
            voice_manager.cleanup_temp_voice(custom_voice['voice_name'])
            logger.info(f"Cleaned up custom voice: {custom_voice['voice_name']}")
        
        # Clean up other temp files
        for key, path in temp_files.items():
            if key in ['transcript', 'scene', 'ref_audio', 'ref_text'] and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.debug(f"Cleaned up: {path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {path}: {e}")
                    
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

def encode_audio_output(output_path: str, output_format: str) -> Dict[str, Any]:
    """Encode the generated audio file to base64"""
    try:
        with open(output_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        # Encode to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Get file size and duration info
        file_size = len(audio_data)
        
        return {
            "audio_base64": audio_base64,
            "sampling_rate": 16000,  # Default for Higgs Audio
            "duration": file_size / (16000 * 2),  # Rough estimate
            "format": output_format,
            "content_type": f"audio/{output_format}",
            "volume_path": output_path,
            "file_size_bytes": file_size
        }
        
    except Exception as e:
        logger.error(f"Error encoding audio output: {e}")
        raise

def check_cache_status() -> Dict[str, Any]:
    """Check the status of cached models"""
    cache_info = {
        "cache_directory": "/runpod-volume/temp_voices",
        "cache_exists": os.path.exists("/runpod-volume/temp_voices"),
        "transformers_cache_exists": os.path.exists("/runpod-volume/temp_voices/transformers"),
        "models_cached": [],
        "all_cached_items": []
    }
    
    transformers_cache_dir = "/runpod-volume/temp_voices/transformers"
    if os.path.exists(transformers_cache_dir):
        try:
            cached_items = os.listdir(transformers_cache_dir)
            cache_info["all_cached_items"] = cached_items
            
            # Look for Higgs Audio related models
            higgs_patterns = ["bosonai", "higgs-audio", "higgs_audio"]
            for item in cached_items:
                item_path = os.path.join(transformers_cache_dir, item)
                if os.path.isdir(item_path):
                    if any(pattern in item.lower() for pattern in higgs_patterns):
                        cache_info["models_cached"].append(item)
                        
                        # Calculate size
                        total_size = 0
                        for root, dirs, files in os.walk(item_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                if os.path.exists(file_path):
                                    total_size += os.path.getsize(file_path)
                        cache_info[f"{item}_size_mb"] = round(total_size / (1024 * 1024), 2)
            
            # Calculate total cache size
            total_cache_size = 0
            for root, dirs, files in os.walk(transformers_cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_cache_size += os.path.getsize(file_path)
            cache_info["total_cache_size_mb"] = round(total_cache_size / (1024 * 1024), 2)
            
        except Exception as e:
            logger.warning(f"Error reading cache directory: {e}")
    
    return cache_info

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced handler function with custom voice support
    
    Features supported:
    - Regular text-to-speech
    - Built-in voice cloning (with ref_audio_base64 + ref_audio_text OR ref_audio_name)
    - Custom user voices from Firebase Storage
    - Long-form chunking (chunk_method: "word", "speaker")
    - Experimental humming ([humming start/end])
    - Experimental BGM ([music start/end] + ref_audio_in_system_message)
    - Scene-based generation
    - Multi-speaker dialogue
    """
    
    temp_files = {}
    
    try:
        logger.info("Processing request with custom voice support...")
        
        # Validate input
        validated_input = validate_input(event)
        logger.info(f"Validated input parameters")
        
        # Create temporary files
        temp_files = create_temp_files(validated_input)
        logger.info(f"Created temporary files: {list(temp_files.keys())}")
        
        # Build command
        cmd = build_generation_command(validated_input, temp_files)
        
        # Run generation
        logger.info(f"🔍 Debug: user_id from validated_input: {validated_input.get('user_id')}")
        logger.info(f"🔍 Debug: ref_audio_name from validated_input: {validated_input.get('ref_audio_name')}")
        logger.info(f"🔍 Debug: About to call run_generation with user_id: {validated_input.get('user_id')}")
        success, output = run_generation(cmd, validated_input["user_id"])
        
        if not success:
            return {
                "error": f"Generation failed: {output}",
                "error_type": "GenerationError"
            }
        
        # Check if output file was created
        output_path = temp_files['output']
        if not os.path.exists(output_path):
            return {
                "error": "Output audio file was not created",
                "error_type": "OutputError"
            }
        
        # Encode audio output
        audio_result = encode_audio_output(output_path, validated_input["output_format"])
        
        # Check cache status
        cache_status = check_cache_status()
        
        # Build response
        response = {
            **audio_result,
            "generated_text": "<|AUDIO_OUT|>",
            "usage": {
                "prompt_tokens": len(validated_input["text"].split()),
                "completion_tokens": validated_input["max_new_tokens"],
                "total_tokens": len(validated_input["text"].split()) + validated_input["max_new_tokens"]
            },
            "method": "subprocess_generation.py_with_custom_voices",
            "command_summary": f"generation.py with {len(cmd)} parameters",
            "cache_status": cache_status
        }
        
        # Add custom voice info if used
        if 'custom_voice' in temp_files:
            response["custom_voice_used"] = {
                "voice_id": validated_input.get("custom_voice_id"),
                "voice_name": temp_files['custom_voice']['voice_name'],
                "user_id": validated_input.get("user_id")
            }
        
        logger.info("Request processed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error in handler: {e}")
        return {
            "error": f"Handler error: {str(e)}",
            "error_type": "HandlerError"
        }
    
    finally:
        # Clean up temporary files
        if temp_files:
            cleanup_temp_files(temp_files)
            logger.info("Temporary files cleaned up")

# Voice management endpoints for SaaS
def handle_voice_upload(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle custom voice upload requests"""
    try:
        input_data = event.get("input", {})
        
        # Validate required fields
        required_fields = ["user_id", "voice_name", "audio_base64", "transcription"]
        for field in required_fields:
            if field not in input_data:
                return {
                    "success": False,
                    "error": f"Missing required field: {field}",
                    "error_type": "ValidationError"
                }
        
        # Validate voice upload
        validation = voice_manager.validate_voice_upload(
            input_data["audio_base64"],
            input_data["transcription"]
        )
        
        if not validation["valid"]:
            return {
                "success": False,
                "error": validation["error"],
                "error_type": "ValidationError"
            }
        
        # Upload voice
        result = voice_manager.upload_custom_voice(
            user_id=input_data["user_id"],
            voice_name=input_data["voice_name"],
            audio_base64=input_data["audio_base64"],
            transcription=input_data["transcription"],
            voice_description=input_data.get("voice_description")
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in voice upload handler: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "HandlerError"
        }

def handle_voice_list(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle voice listing requests"""
    try:
        input_data = event.get("input", {})
        user_id = input_data.get("user_id")
        
        if not user_id:
            return {
                "success": False,
                "error": "Missing user_id",
                "error_type": "ValidationError"
            }
        
        voices = voice_manager.get_user_voices(user_id)
        
        return {
            "success": True,
            "voices": voices,
            "count": len(voices)
        }
        
    except Exception as e:
        logger.error(f"Error in voice list handler: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "HandlerError"
        }

def handle_voice_delete(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle voice deletion requests"""
    try:
        input_data = event.get("input", {})
        
        # Validate required fields
        required_fields = ["user_id", "voice_id"]
        for field in required_fields:
            if field not in input_data:
                return {
                    "success": False,
                    "error": f"Missing required field: {field}",
                    "error_type": "ValidationError"
                }
        
        # Delete voice
        success = voice_manager.delete_voice(
            input_data["voice_id"],
            input_data["user_id"]
        )
        
        if success:
            return {
                "success": True,
                "message": "Voice deleted successfully"
            }
        else:
            return {
                "success": False,
                "error": "Failed to delete voice",
                "error_type": "DeletionError"
            }
        
    except Exception as e:
        logger.error(f"Error in voice delete handler: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "HandlerError"
        }
