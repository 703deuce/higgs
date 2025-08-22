"""
RunPod Serverless Handler for Higgs Audio v2
Provides a serverless API endpoint for audio generation using Higgs Audio v2 model.
"""

import os
import sys
import base64
import torch
import numpy as np
import soundfile as sf
from io import BytesIO
from typing import Dict, Any, Optional, List
import json
import traceback
from loguru import logger

# Add the app directory to Python path to ensure imports work
sys.path.insert(0, '/app')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Higgs Audio imports
try:
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
    from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
except ImportError as e:
    logger.error(f"Failed to import Higgs Audio modules: {e}")
    logger.error("Available paths:")
    for path in sys.path:
        logger.error(f"  - {path}")
    raise

# Global variables for model persistence across invocations
MODEL_ENGINE = None
MODEL_INITIALIZED = False

# Default model configurations
DEFAULT_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
DEFAULT_AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_model():
    """Initialize the Higgs Audio model engine. Called once during cold start."""
    global MODEL_ENGINE, MODEL_INITIALIZED
    
    if MODEL_INITIALIZED:
        return MODEL_ENGINE
    
    try:
        logger.info("Initializing Higgs Audio v2 model...")
        
        # Ensure cache directories exist on RunPod volume
        cache_dirs = [
            "/runpod-volume/cache/huggingface",
            "/runpod-volume/cache/transformers", 
            "/runpod-volume/cache/torch"
        ]
        for cache_dir in cache_dirs:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Cache directory ready: {cache_dir}")
        
        # Get model configuration from environment variables
        model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        audio_tokenizer_path = os.getenv("AUDIO_TOKENIZER_PATH", DEFAULT_AUDIO_TOKENIZER_PATH)
        device = os.getenv("DEVICE", DEFAULT_DEVICE)
        
        logger.info(f"Loading model: {model_path}")
        logger.info(f"Audio tokenizer: {audio_tokenizer_path}")
        logger.info(f"Device: {device}")
        
        # Initialize the serve engine
        MODEL_ENGINE = HiggsAudioServeEngine(
            model_name_or_path=model_path,
            audio_tokenizer_name_or_path=audio_tokenizer_path,
            device=device,
            torch_dtype=torch.bfloat16,
            kv_cache_lengths=[1024, 4096, 8192]  # Multiple KV cache sizes for efficiency
        )
        
        MODEL_INITIALIZED = True
        logger.info("Model initialization completed successfully")
        return MODEL_ENGINE
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

def validate_input(event: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and parse input parameters."""
    
    # Required parameters
    if "input" not in event:
        raise ValueError("Missing 'input' field in request")
    
    input_data = event["input"]
    
    # Extract text input
    if "text" not in input_data:
        raise ValueError("Missing 'text' field in input")
    
    text = input_data["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string")
    
    # Optional parameters with defaults
    validated_input = {
        "text": text.strip(),
        "max_new_tokens": input_data.get("max_new_tokens", 512),
        "temperature": input_data.get("temperature", 0.7),
        "top_k": input_data.get("top_k", 50),
        "top_p": input_data.get("top_p", 0.95),
        "seed": input_data.get("seed", None),
        "force_audio_gen": input_data.get("force_audio_gen", True),
        "ras_win_len": input_data.get("ras_win_len", 7),
        "ras_win_max_num_repeat": input_data.get("ras_win_max_num_repeat", 2),
        "system_prompt": input_data.get("system_prompt", "Generate audio following instruction."),
        "output_format": input_data.get("output_format", "wav"),  # wav, mp3, or base64
        "ref_audio_base64": input_data.get("ref_audio_base64", None),  # Reference audio for voice cloning
        "scene_description": input_data.get("scene_description", None),  # Scene description for context
        # Long-form generation chunking parameters
        "chunk_method": input_data.get("chunk_method", None),  # "word", "speaker", or None
        "chunk_max_word_num": input_data.get("chunk_max_word_num", 200),  # Max words per chunk
        "chunk_max_num_turns": input_data.get("chunk_max_num_turns", 1),  # Max turns per chunk
        "generation_chunk_buffer_size": input_data.get("generation_chunk_buffer_size", None)  # Buffer size for chunks
    }
    
    # Validate numeric parameters
    if not isinstance(validated_input["max_new_tokens"], int) or validated_input["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    
    if not 0.0 <= validated_input["temperature"] <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0")
    
    if not 0.0 <= validated_input["top_p"] <= 1.0:
        raise ValueError("top_p must be between 0.0 and 1.0")
    
    if validated_input["top_k"] is not None and (not isinstance(validated_input["top_k"], int) or validated_input["top_k"] <= 0):
        raise ValueError("top_k must be a positive integer or None")
    
    if validated_input["output_format"] not in ["wav", "mp3", "base64"]:
        raise ValueError("output_format must be 'wav', 'mp3', or 'base64'")
    
    return validated_input

def prepare_messages(validated_input: Dict[str, Any]) -> ChatMLSample:
    """Prepare ChatML messages for the model."""
    
    messages = []
    
    # Add system message
    system_content = validated_input["system_prompt"]
    if validated_input["scene_description"]:
        system_content += f"\n\n<|scene_desc_start|>\n{validated_input['scene_description']}\n<|scene_desc_end|>"
    
    # Handle reference audio if provided
    if validated_input["ref_audio_base64"]:
        # Add audio content to system message
        system_message = Message(
            role="system",
            content=[
                TextContent(text=system_content),
                AudioContent(audio_url="", raw_audio=validated_input["ref_audio_base64"])
            ]
        )
    else:
        system_message = Message(
            role="system",
            content=system_content
        )
    
    messages.append(system_message)
    
    # Add user message with text to generate
    user_message = Message(
        role="user",
        content=validated_input["text"]
    )
    messages.append(user_message)
    
    return ChatMLSample(messages=messages)

def encode_audio_output(audio_data: np.ndarray, sampling_rate: int, output_format: str) -> Dict[str, Any]:
    """Encode audio output in the requested format."""
    
    if audio_data is None:
        return {
            "audio_base64": None,
            "sampling_rate": sampling_rate,
            "duration": 0.0,
            "format": output_format
        }
    
    # Convert to appropriate format
    buffer = BytesIO()
    
    if output_format == "wav":
        sf.write(buffer, audio_data, sampling_rate, format='WAV')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        content_type = "audio/wav"
        
    elif output_format == "mp3":
        # Note: soundfile doesn't support MP3 directly, so we'll use WAV
        # In production, you might want to use pydub for MP3 conversion
        sf.write(buffer, audio_data, sampling_rate, format='WAV')
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        content_type = "audio/wav"  # fallback to WAV
        logger.warning("MP3 format requested but not supported, returning WAV")
        
    else:  # base64 raw
        # Return raw audio data as base64
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        content_type = "audio/raw"
    
    duration = len(audio_data) / sampling_rate
    
    return {
        "audio_base64": audio_base64,
        "sampling_rate": sampling_rate,
        "duration": duration,
        "format": output_format,
        "content_type": content_type
    }

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main handler function for RunPod serverless deployment.
    
    Expected input format:
    {
        "input": {
            "text": "Text to convert to speech",
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "seed": null,
            "force_audio_gen": true,
            "system_prompt": "Generate audio following instruction.",
            "output_format": "wav",
            "ref_audio_base64": null,
            "scene_description": null,
            "chunk_method": null,
            "chunk_max_word_num": 200,
            "chunk_max_num_turns": 1,
            "generation_chunk_buffer_size": null
        }
    }
    
    Returns:
    {
        "audio_base64": "base64_encoded_audio",
        "sampling_rate": 24000,
        "duration": 5.2,
        "format": "wav",
        "generated_text": "Generated text response",
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 200,
            "total_tokens": 250
        }
    }
    """
    
    try:
        logger.info("Processing request...")
        
        # Initialize model if not already done
        model_engine = initialize_model()
        
        # Validate input
        validated_input = validate_input(event)
        logger.info(f"Validated input: {json.dumps({k: v for k, v in validated_input.items() if k != 'ref_audio_base64'})}")
        
        # Prepare ChatML messages
        chat_ml_sample = prepare_messages(validated_input)
        
        # Generate audio
        logger.info("Generating audio...")
        response: HiggsAudioResponse = model_engine.generate(
            chat_ml_sample=chat_ml_sample,
            max_new_tokens=validated_input["max_new_tokens"],
            temperature=validated_input["temperature"],
            top_k=validated_input["top_k"],
            top_p=validated_input["top_p"],
            force_audio_gen=validated_input["force_audio_gen"],
            ras_win_len=validated_input["ras_win_len"],
            ras_win_max_num_repeat=validated_input["ras_win_max_num_repeat"],
            seed=validated_input["seed"],
            stop_strings=["<|end_of_text|>", "<|eot_id|>"]
        )
        
        # Encode audio output
        audio_output = encode_audio_output(
            response.audio, 
            response.sampling_rate, 
            validated_input["output_format"]
        )
        
        # ALSO save audio to network volume for easy access
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            volume_filename = f"/runpod-volume/generated_audio_{timestamp}.wav"
            
            if response.audio is not None:
                # Save audio to volume as WAV file
                import soundfile as sf
                sf.write(volume_filename, response.audio, response.sampling_rate)
                logger.info(f"Audio also saved to volume: {volume_filename}")
                audio_output["volume_file"] = volume_filename
                audio_output["access_note"] = "Audio saved to /runpod-volume/ for easy access"
        except Exception as e:
            logger.warning(f"Failed to save to volume: {e}")

        # Prepare final response
        result = {
            **audio_output,
            "generated_text": response.generated_text,
            "usage": response.usage or {}
        }
        
        logger.info(f"Generation completed successfully. Duration: {audio_output['duration']:.2f}s")
        return result
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "error": error_msg,
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()
        }

# For local testing
import runpod

if __name__ == "__main__":
    # Start the RunPod serverless handler
    runpod.serverless.start({"handler": handler})
