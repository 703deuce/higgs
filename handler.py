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
from typing import Dict, Any
import soundfile as sf

# Add the app directory to Python path for imports
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runpod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_input(event: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize input parameters"""
    if "input" not in event:
        raise ValueError("Missing 'input' field in request")
    
    input_data = event["input"]
    
    if "text" not in input_data:
        raise ValueError("Missing 'text' field in input")
    
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
        "ref_audio_name": input_data.get("ref_audio_name", None),  # Allow direct reference to existing voice samples
        
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
    
    # Option 1: Use existing voice sample by name
    if validated_input["ref_audio_name"]:
        ref_audio_name = validated_input["ref_audio_name"]
        logger.info(f"Using existing voice sample: {ref_audio_name}")
    
    # Option 2: Create temporary voice sample from base64
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
    
    return cmd

def run_generation(cmd: list) -> tuple:
    """Run the generation.py command and return success status and output"""
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for long-form generation
            cwd="/app"
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
    - Voice cloning (with ref_audio_base64 + ref_audio_text OR ref_audio_name)
    - Long-form chunking (chunk_method: "word", "speaker")
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
            "command_summary": f"generation.py with {len(cmd)} parameters"
        }
        
        logger.info(f"Successfully generated audio: {audio_result['duration']}s at {audio_result['sampling_rate']}Hz")
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