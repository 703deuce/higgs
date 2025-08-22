"""
Optimized RunPod Serverless Handler for Higgs Audio v2
Enhanced version with cold start optimization and better error handling.
"""

import os
import base64
import torch
import numpy as np
import soundfile as sf
from io import BytesIO
from typing import Dict, Any, Optional, List
import json
import traceback
from loguru import logger
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Higgs Audio imports
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent

# Global variables for model persistence and optimization
MODEL_ENGINE = None
MODEL_INITIALIZED = False
MODEL_LOCK = threading.Lock()
WARMUP_COMPLETED = False

# Default model configurations
DEFAULT_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
DEFAULT_AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optimization settings
ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "false").lower() == "true"
ENABLE_TENSOR_PARALLEL = os.getenv("ENABLE_TENSOR_PARALLEL", "false").lower() == "true"
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "1"))

def optimize_torch_settings():
    """Optimize PyTorch settings for inference."""
    try:
        # Set optimal number of threads
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Set memory fraction to avoid OOM
            torch.cuda.set_per_process_memory_fraction(0.9)
            
        logger.info("PyTorch optimization settings applied")
    except Exception as e:
        logger.warning(f"Failed to apply some PyTorch optimizations: {e}")

def download_models_parallel():
    """Pre-download models in parallel during container startup."""
    try:
        from transformers import AutoTokenizer, AutoConfig
        from huggingface_hub import hf_hub_download
        
        model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        audio_tokenizer_path = os.getenv("AUDIO_TOKENIZER_PATH", DEFAULT_AUDIO_TOKENIZER_PATH)
        
        def download_model():
            try:
                logger.info(f"Pre-downloading model: {model_path}")
                AutoConfig.from_pretrained(model_path)
                AutoTokenizer.from_pretrained(model_path)
                logger.info("Model download completed")
            except Exception as e:
                logger.warning(f"Model pre-download failed: {e}")
        
        def download_tokenizer():
            try:
                logger.info(f"Pre-downloading audio tokenizer: {audio_tokenizer_path}")
                # This will trigger the download without full loading
                AutoConfig.from_pretrained(audio_tokenizer_path)
                logger.info("Audio tokenizer download completed")
            except Exception as e:
                logger.warning(f"Audio tokenizer pre-download failed: {e}")
        
        # Download in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(download_model)
            executor.submit(download_tokenizer)
            
    except Exception as e:
        logger.warning(f"Parallel download setup failed: {e}")

def initialize_model():
    """Initialize the Higgs Audio model engine with optimizations."""
    global MODEL_ENGINE, MODEL_INITIALIZED
    
    with MODEL_LOCK:
        if MODEL_INITIALIZED:
            return MODEL_ENGINE
        
        try:
            logger.info("Initializing Higgs Audio v2 model with optimizations...")
            start_time = time.time()
            
            # Apply PyTorch optimizations
            optimize_torch_settings()
            
            # Get model configuration from environment variables
            model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
            audio_tokenizer_path = os.getenv("AUDIO_TOKENIZER_PATH", DEFAULT_AUDIO_TOKENIZER_PATH)
            device = os.getenv("DEVICE", DEFAULT_DEVICE)
            
            logger.info(f"Loading model: {model_path}")
            logger.info(f"Audio tokenizer: {audio_tokenizer_path}")
            logger.info(f"Device: {device}")
            
            # Determine optimal cache lengths based on available memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                if gpu_memory >= 80:  # A100 or similar
                    kv_cache_lengths = [1024, 4096, 8192, 16384]
                elif gpu_memory >= 40:  # A40 or similar
                    kv_cache_lengths = [1024, 4096, 8192]
                else:  # Smaller GPUs
                    kv_cache_lengths = [1024, 4096]
            else:
                kv_cache_lengths = [1024, 2048]
            
            logger.info(f"Using KV cache lengths: {kv_cache_lengths}")
            
            # Initialize the serve engine
            MODEL_ENGINE = HiggsAudioServeEngine(
                model_name_or_path=model_path,
                audio_tokenizer_name_or_path=audio_tokenizer_path,
                device=device,
                torch_dtype=torch.bfloat16,
                kv_cache_lengths=kv_cache_lengths
            )
            
            # Apply model optimizations
            if ENABLE_TORCH_COMPILE and hasattr(torch, 'compile'):
                try:
                    logger.info("Compiling model with torch.compile...")
                    MODEL_ENGINE.model = torch.compile(MODEL_ENGINE.model, mode="reduce-overhead")
                    logger.info("Model compilation completed")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
            
            MODEL_INITIALIZED = True
            init_time = time.time() - start_time
            logger.info(f"Model initialization completed successfully in {init_time:.2f}s")
            
            return MODEL_ENGINE
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            logger.error(traceback.format_exc())
            raise e

def warmup_model():
    """Warm up the model with a test inference to reduce first-request latency."""
    global WARMUP_COMPLETED
    
    if WARMUP_COMPLETED:
        return
    
    try:
        logger.info("Starting model warmup...")
        start_time = time.time()
        
        model_engine = initialize_model()
        
        # Create a simple test input
        test_messages = [
            Message(role="system", content="Generate audio following instruction."),
            Message(role="user", content="Hello, this is a test.")
        ]
        test_sample = ChatMLSample(messages=test_messages)
        
        # Run a short generation to warm up
        _ = model_engine.generate(
            chat_ml_sample=test_sample,
            max_new_tokens=32,
            temperature=0.7,
            force_audio_gen=True
        )
        
        WARMUP_COMPLETED = True
        warmup_time = time.time() - start_time
        logger.info(f"Model warmup completed in {warmup_time:.2f}s")
        
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")

def validate_input(event: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and parse input parameters with enhanced validation."""
    
    # Check if input is provided
    if "input" not in event:
        raise ValueError("Missing 'input' field in request")
    
    input_data = event["input"]
    
    # Extract and validate text input
    if "text" not in input_data:
        raise ValueError("Missing 'text' field in input")
    
    text = input_data["text"]
    if not isinstance(text, str):
        raise ValueError("Text must be a string")
    
    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty")
    
    # Check text length
    max_text_length = int(os.getenv("MAX_TEXT_LENGTH", "10000"))
    if len(text) > max_text_length:
        raise ValueError(f"Text length ({len(text)}) exceeds maximum allowed ({max_text_length})")
    
    # Optional parameters with validation and defaults
    validated_input = {
        "text": text,
        "max_new_tokens": min(input_data.get("max_new_tokens", 512), 2048),  # Cap at 2048
        "temperature": max(0.0, min(input_data.get("temperature", 0.7), 2.0)),
        "top_k": input_data.get("top_k", 50),
        "top_p": max(0.0, min(input_data.get("top_p", 0.95), 1.0)),
        "seed": input_data.get("seed", None),
        "force_audio_gen": input_data.get("force_audio_gen", True),
        "ras_win_len": max(0, input_data.get("ras_win_len", 7)) if input_data.get("ras_win_len") is not None else 7,
        "ras_win_max_num_repeat": max(1, input_data.get("ras_win_max_num_repeat", 2)),
        "system_prompt": input_data.get("system_prompt", "Generate audio following instruction."),
        "output_format": input_data.get("output_format", "wav"),
        "ref_audio_base64": input_data.get("ref_audio_base64", None),
        "scene_description": input_data.get("scene_description", None)
    }
    
    # Validate specific parameters
    if validated_input["top_k"] is not None and validated_input["top_k"] <= 0:
        validated_input["top_k"] = None
    
    if validated_input["output_format"] not in ["wav", "mp3", "base64"]:
        raise ValueError("output_format must be 'wav', 'mp3', or 'base64'")
    
    if validated_input["seed"] is not None and not isinstance(validated_input["seed"], int):
        raise ValueError("seed must be an integer or None")
    
    # Validate reference audio if provided
    if validated_input["ref_audio_base64"]:
        try:
            base64.b64decode(validated_input["ref_audio_base64"])
        except Exception:
            raise ValueError("Invalid base64 encoding for reference audio")
    
    return validated_input

def prepare_messages(validated_input: Dict[str, Any]) -> ChatMLSample:
    """Prepare ChatML messages for the model with better context handling."""
    
    messages = []
    
    # Build system message with context
    system_parts = [validated_input["system_prompt"]]
    
    if validated_input["scene_description"]:
        system_parts.append(f"<|scene_desc_start|>\n{validated_input['scene_description']}\n<|scene_desc_end|>")
    
    system_content = "\n\n".join(system_parts)
    
    # Handle reference audio if provided
    if validated_input["ref_audio_base64"]:
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
    
    # Add user message
    user_message = Message(
        role="user",
        content=validated_input["text"]
    )
    messages.append(user_message)
    
    return ChatMLSample(messages=messages)

def encode_audio_output(audio_data: np.ndarray, sampling_rate: int, output_format: str) -> Dict[str, Any]:
    """Encode audio output with better error handling."""
    
    if audio_data is None:
        return {
            "audio_base64": None,
            "sampling_rate": sampling_rate,
            "duration": 0.0,
            "format": output_format,
            "content_type": "audio/wav"
        }
    
    try:
        buffer = BytesIO()
        
        # Ensure audio is in the right format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalize if needed
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        if output_format in ["wav", "mp3"]:  # Use WAV for both (MP3 conversion would need additional libraries)
            sf.write(buffer, audio_data, sampling_rate, format='WAV', subtype='PCM_16')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            content_type = "audio/wav"
            
        else:  # base64 raw
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
        
    except Exception as e:
        logger.error(f"Audio encoding failed: {e}")
        raise ValueError(f"Failed to encode audio: {str(e)}")

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimized main handler function for RunPod serverless deployment.
    """
    
    request_start_time = time.time()
    
    try:
        logger.info("Processing request...")
        
        # Initialize and warm up model if needed
        model_engine = initialize_model()
        if not WARMUP_COMPLETED:
            warmup_model()
        
        # Validate input
        validated_input = validate_input(event)
        logger.info(f"Processing text: {validated_input['text'][:100]}{'...' if len(validated_input['text']) > 100 else ''}")
        
        # Prepare ChatML messages
        chat_ml_sample = prepare_messages(validated_input)
        
        # Generate audio
        logger.info("Generating audio...")
        generation_start = time.time()
        
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
        
        generation_time = time.time() - generation_start
        
        # Encode audio output
        audio_output = encode_audio_output(
            response.audio, 
            response.sampling_rate, 
            validated_input["output_format"]
        )
        
        # Prepare final response with timing information
        total_time = time.time() - request_start_time
        
        result = {
            **audio_output,
            "generated_text": response.generated_text,
            "usage": response.usage or {},
            "timing": {
                "total_time": round(total_time, 3),
                "generation_time": round(generation_time, 3),
                "audio_duration": audio_output["duration"]
            },
            "model_info": {
                "model_path": os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH),
                "device": DEFAULT_DEVICE,
                "warmup_completed": WARMUP_COMPLETED
            }
        }
        
        logger.info(f"Generation completed in {generation_time:.2f}s (total: {total_time:.2f}s). Audio duration: {audio_output['duration']:.2f}s")
        return result
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        return {
            "error": error_msg,
            "error_type": type(e).__name__,
            "timing": {
                "total_time": round(time.time() - request_start_time, 3)
            }
        }

# Pre-download models when module is imported (container startup)
if os.getenv("PRELOAD_MODELS", "true").lower() == "true":
    try:
        download_models_parallel()
    except Exception as e:
        logger.warning(f"Model pre-loading failed: {e}")

# For local testing
if __name__ == "__main__":
    # Test the optimized handler locally
    test_event = {
        "input": {
            "text": "Hello, this is a test of the optimized Higgs Audio v2 model running on RunPod!",
            "temperature": 0.3,
            "max_new_tokens": 256,
            "output_format": "wav"
        }
    }
    
    print("Testing optimized handler...")
    result = handler(test_event)
    print(json.dumps({k: v for k, v in result.items() if k != 'audio_base64'}, indent=2))
