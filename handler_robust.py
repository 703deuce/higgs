"""
Robust RunPod Serverless Handler for Higgs Audio v2
This version handles import issues more gracefully and provides better error diagnostics.
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

# Ensure Python can find our modules
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if '/app' not in sys.path:
    sys.path.insert(0, '/app')

# Global variables
MODEL_ENGINE = None
MODEL_INITIALIZED = False
IMPORT_ERROR = None

def setup_imports():
    """Handle imports with detailed error reporting."""
    global IMPORT_ERROR
    
    try:
        # Try to import the required modules
        logger.info("Setting up Higgs Audio imports...")
        logger.info(f"Python path: {sys.path}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"App directory: {APP_DIR}")
        
        # Check if the boson_multimodal directory exists
        boson_path = os.path.join(APP_DIR, 'boson_multimodal')
        logger.info(f"Looking for boson_multimodal at: {boson_path}")
        logger.info(f"boson_multimodal exists: {os.path.exists(boson_path)}")
        
        if os.path.exists(boson_path):
            contents = os.listdir(boson_path)
            logger.info(f"boson_multimodal contents: {contents}")
        
        # Try imports step by step
        logger.info("Step 1: Importing boson_multimodal...")
        import boson_multimodal
        logger.info("✅ boson_multimodal imported successfully")
        
        logger.info("Step 2: Importing data_types...")
        from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
        logger.info("✅ data_types imported successfully")
        
        logger.info("Step 3: Importing serve_engine...")
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
        logger.info("✅ serve_engine imported successfully")
        
        return {
            'HiggsAudioServeEngine': HiggsAudioServeEngine,
            'HiggsAudioResponse': HiggsAudioResponse,
            'Message': Message,
            'ChatMLSample': ChatMLSample,
            'AudioContent': AudioContent,
            'TextContent': TextContent
        }
        
    except Exception as e:
        IMPORT_ERROR = str(e)
        logger.error(f"❌ Import failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Try to provide helpful diagnostics
        logger.error("=== DIAGNOSTIC INFORMATION ===")
        logger.error(f"Python version: {sys.version}")
        logger.error(f"Python path: {sys.path}")
        logger.error(f"Working directory: {os.getcwd()}")
        logger.error(f"Environment variables:")
        for key, value in os.environ.items():
            if 'PYTHON' in key or 'PATH' in key:
                logger.error(f"  {key}: {value}")
        
        # Check if specific files exist
        critical_files = [
            'boson_multimodal/__init__.py',
            'boson_multimodal/serve/serve_engine.py',
            'boson_multimodal/data_types.py',
            'boson_multimodal/model/higgs_audio/__init__.py',
            'boson_multimodal/model/higgs_audio/modeling_higgs_audio.py'
        ]
        
        logger.error("Critical files check:")
        for file_path in critical_files:
            full_path = os.path.join(APP_DIR, file_path)
            exists = os.path.exists(full_path)
            logger.error(f"  {file_path}: {'✅' if exists else '❌'}")
        
        return None

# Setup imports at module level
logger.info("Loading Higgs Audio modules...")
MODULES = setup_imports()

def initialize_model():
    """Initialize the Higgs Audio model engine."""
    global MODEL_ENGINE, MODEL_INITIALIZED, MODULES
    
    if IMPORT_ERROR:
        raise ImportError(f"Cannot initialize model due to import error: {IMPORT_ERROR}")
    
    if not MODULES:
        raise ImportError("Required modules not available")
    
    if MODEL_INITIALIZED:
        return MODEL_ENGINE
    
    try:
        logger.info("Initializing Higgs Audio v2 model...")
        
        # Get model configuration from environment variables
        model_path = os.getenv("MODEL_PATH", "bosonai/higgs-audio-v2-generation-3B-base")
        audio_tokenizer_path = os.getenv("AUDIO_TOKENIZER_PATH", "bosonai/higgs-audio-v2-tokenizer")
        device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model: {model_path}")
        logger.info(f"Audio tokenizer: {audio_tokenizer_path}")
        logger.info(f"Device: {device}")
        
        # Initialize the serve engine
        HiggsAudioServeEngine = MODULES['HiggsAudioServeEngine']
        MODEL_ENGINE = HiggsAudioServeEngine(
            model_name_or_path=model_path,
            audio_tokenizer_name_or_path=audio_tokenizer_path,
            device=device,
            torch_dtype=torch.bfloat16,
            kv_cache_lengths=[1024, 4096, 8192]
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
    
    if "input" not in event:
        raise ValueError("Missing 'input' field in request")
    
    input_data = event["input"]
    
    if "text" not in input_data:
        raise ValueError("Missing 'text' field in input")
    
    text = input_data["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string")
    
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
        "output_format": input_data.get("output_format", "wav"),
        "ref_audio_base64": input_data.get("ref_audio_base64", None),
        "scene_description": input_data.get("scene_description", None)
    }
    
    # Validate parameters
    if not isinstance(validated_input["max_new_tokens"], int) or validated_input["max_new_tokens"] <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    
    if not 0.0 <= validated_input["temperature"] <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0")
    
    if not 0.0 <= validated_input["top_p"] <= 1.0:
        raise ValueError("top_p must be between 0.0 and 1.0")
    
    if validated_input["output_format"] not in ["wav", "mp3", "base64"]:
        raise ValueError("output_format must be 'wav', 'mp3', or 'base64'")
    
    return validated_input

def prepare_messages(validated_input: Dict[str, Any]):
    """Prepare ChatML messages for the model."""
    
    if not MODULES:
        raise ImportError("Required modules not available")
    
    Message = MODULES['Message']
    ChatMLSample = MODULES['ChatMLSample']
    AudioContent = MODULES['AudioContent']
    TextContent = MODULES['TextContent']
    
    messages = []
    
    # Add system message
    system_content = validated_input["system_prompt"]
    if validated_input["scene_description"]:
        system_content += f"\n\n<|scene_desc_start|>\n{validated_input['scene_description']}\n<|scene_desc_end|>"
    
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
    """Encode audio output in the requested format."""
    
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
        
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        if output_format in ["wav", "mp3"]:
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
    """Main handler function for RunPod serverless deployment."""
    
    try:
        logger.info("Processing request...")
        
        # Check for import errors first
        if IMPORT_ERROR:
            return {
                "error": f"Import error: {IMPORT_ERROR}",
                "error_type": "ImportError",
                "diagnostic_info": {
                    "python_version": sys.version,
                    "working_directory": os.getcwd(),
                    "python_path": sys.path[:3],  # First 3 entries
                    "app_directory": APP_DIR
                }
            }
        
        if not MODULES:
            return {
                "error": "Required modules not available",
                "error_type": "ImportError"
            }
        
        # Initialize model
        model_engine = initialize_model()
        
        # Validate input
        validated_input = validate_input(event)
        logger.info(f"Processing text: {validated_input['text'][:100]}{'...' if len(validated_input['text']) > 100 else ''}")
        
        # Prepare ChatML messages
        chat_ml_sample = prepare_messages(validated_input)
        
        # Generate audio
        logger.info("Generating audio...")
        HiggsAudioResponse = MODULES['HiggsAudioResponse']
        
        response = model_engine.generate(
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
        
        # Prepare final response
        result = {
            **audio_output,
            "generated_text": response.generated_text,
            "usage": response.usage or {},
            "model_info": {
                "model_initialized": MODEL_INITIALIZED,
                "import_success": IMPORT_ERROR is None
            }
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
if __name__ == "__main__":
    print("Testing robust handler...")
    test_event = {
        "input": {
            "text": "Hello, this is a test of the robust Higgs Audio handler!",
            "temperature": 0.3,
            "max_new_tokens": 128
        }
    }
    
    result = handler(test_event)
    print(json.dumps(result, indent=2))
