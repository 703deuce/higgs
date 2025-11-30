#!/usr/bin/env python3
"""
Handler that uses subprocess to call the official generation.py script.
This ensures we use the exact same logic as the command-line tool for ALL features.
"""

import os
import sys
import json
import base64
import tempfile
import subprocess
import logging
import time
import shutil
from typing import Dict, Any
import soundfile as sf

# Set up Hugging Face cache to use persistent volume BEFORE any imports
os.environ["HF_HOME"] = "/runpod-volume/.huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume/.huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/runpod-volume/.huggingface/datasets"
os.environ["TORCH_HOME"] = "/runpod-volume/.torch"

# Create cache directories if they don't exist
os.makedirs("/runpod-volume/.huggingface/transformers", exist_ok=True)
os.makedirs("/runpod-volume/.huggingface/datasets", exist_ok=True)
os.makedirs("/runpod-volume/.torch", exist_ok=True)

# Add the app directory to Python path for imports
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runpod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log cache configuration
logger.info(f"HF_HOME set to: {os.environ.get('HF_HOME')}")
logger.info(f"TRANSFORMERS_CACHE set to: {os.environ.get('TRANSFORMERS_CACHE')}")
logger.info(f"Cache directories created and ready for model persistence")

def validate_input(event: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize input parameters"""
    if "input" not in event:
        raise ValueError("Missing 'input' field in request")
    
    input_data = event["input"]
    
    if "text" not in input_data:
        raise ValueError("Missing 'text' field in input")
    
    # Check if user_id is required (only for custom voices by name)
    ref_audio_name = input_data.get("ref_audio_name")
    ref_audio_firebase_path = input_data.get("ref_audio_firebase_path")
    
    # Validate that only one ref_audio method is used
    ref_audio_methods = [
        bool(ref_audio_name),
        bool(ref_audio_firebase_path),
        bool(input_data.get("ref_audio_base64"))
    ]
    if sum(ref_audio_methods) > 1:
        raise ValueError("Only one of 'ref_audio_name', 'ref_audio_firebase_path', or 'ref_audio_base64' can be provided")
    
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
        "ref_audio_name": ref_audio_name,  # Allow direct reference to existing voice samples
        "ref_audio_firebase_path": ref_audio_firebase_path,  # Firebase Storage path or URL
        "ref_audio_firebase_text_path": input_data.get("ref_audio_firebase_text_path", None),  # Optional text path
        
        # Custom voice support
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
    """Create temporary files needed for generation.py"""
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
    
    # Handle reference audio
    ref_audio_name = None
    
    # Option 1: Download from Firebase Storage path/URL
    if validated_input["ref_audio_firebase_path"]:
        try:
            # Import voice manager
            sys.path.append('/app')
            from voice_management import VoiceManager
            
            # Firebase configuration
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
            
            # Download from Firebase path
            logger.info(f"Downloading voice from Firebase path: {validated_input['ref_audio_firebase_path']}")
            voice_info = voice_manager.download_from_firebase_path(
                validated_input["ref_audio_firebase_path"],
                validated_input.get("ref_audio_firebase_text_path")
            )
            
            if voice_info and os.path.exists(voice_info['audio_path']):
                ref_audio_name = voice_info['voice_name']
                
                # Copy files to voice_prompts directory so generation.py can find them
                # This is where generation.py expects to find voice samples
                voice_prompts_dir = "/app/examples/voice_prompts"
                os.makedirs(voice_prompts_dir, exist_ok=True)
                
                final_audio_path = os.path.join(voice_prompts_dir, f"{ref_audio_name}.wav")
                final_text_path = os.path.join(voice_prompts_dir, f"{ref_audio_name}.txt")
                
                # Copy audio file
                shutil.copy2(voice_info['audio_path'], final_audio_path)
                temp_files['ref_audio'] = final_audio_path
                
                # Copy text file if it exists
                if os.path.exists(voice_info['text_path']) and os.path.getsize(voice_info['text_path']) > 0:
                    shutil.copy2(voice_info['text_path'], final_text_path)
                    temp_files['ref_text'] = final_text_path
                else:
                    # Create empty text file as fallback
                    with open(final_text_path, 'w', encoding='utf-8') as f:
                        f.write("")
                    temp_files['ref_text'] = final_text_path
                
                logger.info(f"✅ Successfully downloaded and prepared voice from Firebase: {ref_audio_name}")
            else:
                raise ValueError(f"Failed to download voice from Firebase path: {validated_input['ref_audio_firebase_path']}")
                
        except Exception as e:
            logger.error(f"Error downloading from Firebase path: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to download voice from Firebase: {str(e)}")
    
    # Option 2: Use existing voice sample by name
    elif validated_input["ref_audio_name"]:
        ref_audio_name = validated_input["ref_audio_name"]
        logger.info(f"Using existing voice sample: {ref_audio_name}")
    
    # Option 3: Create temporary voice sample from base64
    elif validated_input["ref_audio_base64"]:
        # Decode base64 audio
        audio_data = base64.b64decode(validated_input["ref_audio_base64"])
        
        # Create a unique name for this reference audio
        ref_audio_name = f"temp_ref_{int(time.time())}"
        
        # Save reference audio in voice_prompts directory (where generation.py expects it)
        voice_prompts_dir = "/app/examples/voice_prompts"
        os.makedirs(voice_prompts_dir, exist_ok=True)
        
        ref_audio_path = os.path.join(voice_prompts_dir, f"{ref_audio_name}.wav")
        with open(ref_audio_path, 'wb') as f:
            f.write(audio_data)
        temp_files['ref_audio'] = ref_audio_path
        
        # Save reference transcription in voice_prompts directory
        if validated_input["ref_audio_text"]:
            ref_text_path = os.path.join(voice_prompts_dir, f"{ref_audio_name}.txt")
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
    """Build the command line arguments for generation.py"""
    
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
    
    # Add user_id if provided (for custom voices)
    if validated_input.get("user_id"):
        cmd.extend(["--user_id", validated_input["user_id"]])
    
    return cmd

def run_generation(cmd: list) -> tuple:
    """Run the generation.py command and return success status and output"""
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Ensure subprocess inherits the cache environment variables
        env = os.environ.copy()
        env.update({
            "HF_HOME": "/runpod-volume/.huggingface",
            "TRANSFORMERS_CACHE": "/runpod-volume/.huggingface/transformers",
            "HF_DATASETS_CACHE": "/runpod-volume/.huggingface/datasets",
            "TORCH_HOME": "/runpod-volume/.torch"
        })
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for long-form generation
            cwd="/app",
            env=env  # Pass the environment with cache variables
        )
        
        if result.returncode == 0:
            logger.info("Generation completed successfully")
            return True, result.stdout
        else:
            logger.error(f"Generation failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        logger.error("Generation timed out after 10 minutes")
        return False, "Generation timed out"
    except Exception as e:
        logger.error(f"Error running generation: {e}")
        return False, str(e)

def check_cache_status() -> Dict[str, Any]:
    """Check the status of the Hugging Face cache"""
    cache_info = {
        "cache_directory": "/runpod-volume/.huggingface",
        "cache_exists": os.path.exists("/runpod-volume/.huggingface"),
        "transformers_cache_exists": os.path.exists("/runpod-volume/.huggingface/transformers"),
        "models_cached": [],
        "all_cached_items": []
    }
    
    transformers_cache_dir = "/runpod-volume/.huggingface/transformers"
    if os.path.exists(transformers_cache_dir):
        try:
            # List all items in the cache directory
            cached_items = os.listdir(transformers_cache_dir)
            cache_info["all_cached_items"] = cached_items
            
            # Look for any Higgs Audio related models (various naming patterns)
            higgs_patterns = ["bosonai", "higgs-audio", "higgs_audio"]
            
            for item in cached_items:
                item_path = os.path.join(transformers_cache_dir, item)
                if os.path.isdir(item_path):
                    # Check if this looks like a Higgs Audio model
                    if any(pattern in item.lower() for pattern in higgs_patterns):
                        cache_info["models_cached"].append(item)
                        
                        # Get cache size
                        try:
                            total_size = sum(
                                os.path.getsize(os.path.join(dirpath, filename))
                                for dirpath, dirnames, filenames in os.walk(item_path)
                                for filename in filenames
                            )
                            cache_info[f"{item}_size_mb"] = round(total_size / (1024 * 1024), 2)
                        except Exception as e:
                            logger.warning(f"Could not calculate size for {item}: {e}")
            
            # Calculate total cache size
            try:
                total_cache_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(transformers_cache_dir)
                    for filename in filenames
                )
                cache_info["total_cache_size_mb"] = round(total_cache_size / (1024 * 1024), 2)
            except Exception as e:
                logger.warning(f"Could not calculate total cache size: {e}")
                
        except Exception as e:
            logger.warning(f"Error reading cache directory: {e}")
    
    return cache_info

def cleanup_temp_files(temp_files: Dict[str, str]):
    """Clean up temporary files"""
    for file_type, file_path in temp_files.items():
        if file_type != 'ref_name' and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.debug(f"Cleaned up {file_type}: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up {file_path}: {e}")

def encode_audio_output(output_path: str, output_format: str) -> Dict[str, Any]:
    """Read generated audio and encode it for the response"""
    try:
        # Read the generated audio
        audio_data, sampling_rate = sf.read(output_path)
        
        # Convert to base64
        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Calculate duration
        duration = len(audio_data) / sampling_rate
        
        # Save to volume for persistence
        timestamp = int(time.time())
        volume_filename = f"/runpod-volume/generated_audio_subprocess_{timestamp}.wav"
        os.makedirs("/runpod-volume", exist_ok=True)
        sf.write(volume_filename, audio_data, sampling_rate)
        logger.info(f"Audio also saved to volume: {volume_filename}")
        
        return {
            "audio_base64": audio_base64,
            "sampling_rate": int(sampling_rate),
            "duration": round(duration, 2),
            "format": output_format,
            "content_type": f"audio/{output_format}",
            "volume_path": volume_filename
        }
        
    except Exception as e:
        logger.error(f"Error encoding audio output: {e}")
        raise

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler function that uses subprocess to call generation.py
    
    This approach ensures we use the exact same logic as the official command-line tool,
    supporting ALL features including chunking, voice cloning, and experimental capabilities.
    
    Features supported:
    - Regular text-to-speech
    - Voice cloning (with ref_audio_base64 + ref_audio_text OR ref_audio_name OR ref_audio_firebase_path)
    - Firebase Storage path/URL support for custom voices (ref_audio_firebase_path)
    - Long-form chunking (chunk_method: "word", "speaker", "sentence", "semantic", "adaptive", "clause")
    - Experimental humming ([humming start/end])
    - Experimental BGM ([music start/end] + ref_audio_in_system_message)
    - Scene-based generation
    - Multi-speaker dialogue
    """
    
    temp_files = {}
    
    try:
        logger.info("Processing request with subprocess approach...")
        
        # Validate input
        validated_input = validate_input(event)
        logger.info(f"Validated input parameters")
        
        # Create temporary files
        temp_files = create_temp_files(validated_input)
        logger.info(f"Created temporary files: {list(temp_files.keys())}")
        
        # Build command
        cmd = build_generation_command(validated_input, temp_files)
        
        # Run generation
        success, output = run_generation(cmd)
        
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
            "generated_text": "<|AUDIO_OUT|>",  # Standard response for audio generation
            "usage": {
                "prompt_tokens": len(validated_input["text"].split()),
                "completion_tokens": validated_input["max_new_tokens"],
                "total_tokens": len(validated_input["text"].split()) + validated_input["max_new_tokens"]
            },
            "method": "subprocess_generation.py",
            "command_summary": f"generation.py with {len(cmd)} parameters",
            "cache_status": cache_status
        }
        
        logger.info(f"Successfully generated audio: {audio_result['duration']}s at {audio_result['sampling_rate']}Hz")
        logger.info(f"Cache status: {len(cache_status['models_cached'])} models cached")
        return response
        
    except Exception as e:
        logger.error(f"Error in handler: {e}")
        return {
            "error": str(e),
            "error_type": type(e).__name__
        }
    finally:
        # Clean up temporary files
        if temp_files:
            cleanup_temp_files(temp_files)

if __name__ == "__main__":
    # Start the RunPod serverless handler
    runpod.serverless.start({"handler": handler})