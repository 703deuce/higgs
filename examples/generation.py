"""Example script for generating audio using HiggsAudio."""

import click
import soundfile as sf
import langid
import jieba
import os
import re
import copy
import torchaudio
import tqdm
import yaml
import nltk
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Optional

from loguru import logger
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent

from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import (
    ChatMLDatasetSample,
    prepare_chatml_sample,
)
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from typing import List
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import StaticCache
from typing import Optional
from dataclasses import asdict
import torch

CURR_DIR = os.path.dirname(os.path.abspath(__file__))


AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"


def ensure_nltk_data():
    """Ensure NLTK data is downloaded for sentence tokenization"""
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        logger.info("Downloading NLTK punkt_tab tokenizer...")
        nltk.download('punkt_tab', quiet=True)
        logger.info("NLTK punkt_tab tokenizer downloaded successfully")
    
    # Also ensure the legacy punkt is available for compatibility
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer (legacy)...")
        nltk.download('punkt', quiet=True)
        logger.info("NLTK punkt tokenizer downloaded successfully")

def ensure_spacy_model():
    """Ensure spaCy English model is downloaded"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        logger.info("Downloading spaCy English model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy English model downloaded successfully")
        return nlp

def get_sentence_transformer():
    """Get sentence transformer model for semantic analysis"""
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        logger.warning(f"Could not load sentence transformer: {e}")
        return None

def calculate_semantic_similarity(sentences: List[str], model) -> np.ndarray:
    """Calculate semantic similarity matrix between sentences"""
    if model is None:
        return np.eye(len(sentences))  # Return identity matrix if no model
    
    try:
        embeddings = model.encode(sentences)
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    except Exception as e:
        logger.warning(f"Error calculating semantic similarity: {e}")
        return np.eye(len(sentences))

def detect_clause_boundaries(text: str, nlp) -> List[int]:
    """Detect clause boundaries using spaCy dependency parsing"""
    if nlp is None:
        # Fallback to simple punctuation-based detection
        boundaries = []
        for i, char in enumerate(text):
            if char in ',;':
                boundaries.append(i)
        return boundaries
    
    try:
        doc = nlp(text)
        boundaries = []
        for token in doc:
            # Detect clause boundaries based on dependency parsing
            if token.dep_ in ['cc', 'punct'] and token.text in ',;':
                boundaries.append(token.idx)
        return boundaries
    except Exception as e:
        logger.warning(f"Error in clause boundary detection: {e}")
        return []

def calculate_content_complexity(sentence: str, nlp) -> float:
    """Calculate content complexity score for adaptive chunking"""
    if nlp is None:
        # Simple fallback: word count and punctuation
        word_count = len(sentence.split())
        punct_count = sum(1 for c in sentence if c in '.,;:!?')
        return word_count + punct_count * 0.5
    
    try:
        doc = nlp(sentence)
        complexity_score = 0
        
        # Factors that increase complexity
        complexity_score += len(doc)  # Word count
        complexity_score += len([t for t in doc if t.dep_ in ['nsubj', 'dobj', 'pobj']]) * 0.5  # Arguments
        complexity_score += len([t for t in doc if t.pos_ in ['ADJ', 'ADV']]) * 0.3  # Modifiers
        complexity_score += len([t for t in doc if t.dep_ == 'cc']) * 0.4  # Conjunctions
        
        return complexity_score
    except Exception as e:
        logger.warning(f"Error calculating content complexity: {e}")
        return len(sentence.split())

def validate_chunk_order_preservation(original_text: str, chunks: List[str]) -> bool:
    """Validate that chunks preserve the original text order"""
    try:
        # Reconstruct text from chunks (remove extra spaces)
        reconstructed = " ".join(chunks)
        original_clean = " ".join(original_text.split())
        
        # Check if all original words appear in the same order
        original_words = original_clean.split()
        reconstructed_words = reconstructed.split()
        
        if len(original_words) != len(reconstructed_words):
            logger.warning(f"Word count mismatch: original={len(original_words)}, reconstructed={len(reconstructed_words)}")
            return False
        
        # Check word order preservation
        for i, (orig_word, recon_word) in enumerate(zip(original_words, reconstructed_words)):
            if orig_word.lower() != recon_word.lower():
                logger.warning(f"Word order violation at position {i}: '{orig_word}' vs '{recon_word}'")
                return False
        
        logger.info("✅ Chunk order preservation validated successfully")
        return True
        
    except Exception as e:
        logger.warning(f"Error validating chunk order: {e}")
        return False


MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


def resolve_voice_paths(character_name, base_dir=None, user_id=None):
    """
    SMART voice path resolver that automatically handles both regular and custom voices:
    1. Normal voice_prompts folder (built-in voices) - PRIMARY LOCATION
    2. Firebase temp location (custom voices) - SECONDARY LOCATION
    3. Automatic Firebase download for custom voices when needed
    4. Fallback to original path for normal error handling
    
    This function preserves ALL existing behavior while adding automatic custom voice support.
    """
    if base_dir is None:
        base_dir = CURR_DIR
    
    # Location 1: Normal voice_prompts (built-in voices) - PRIMARY
    normal_audio_path = os.path.join(f"{base_dir}/voice_prompts", f"{character_name}.wav")
    normal_text_path = os.path.join(f"{base_dir}/voice_prompts", f"{character_name}.txt")
    
    # Check if normal paths exist (built-in voices)
    if os.path.exists(normal_audio_path) and os.path.exists(normal_text_path):
        logger.info(f"🎤 Using regular voice: {character_name}")
        return normal_audio_path, normal_text_path
    
    # Location 2: Firebase temp location (custom voices) - SECONDARY
    firebase_audio_path = os.path.join("/runpod-volume/temp_voices", f"{character_name}.wav")
    firebase_text_path = os.path.join("/runpod-volume/temp_voices", f"{character_name}.txt")
    
    # Check if Firebase paths exist (custom voices)
    if os.path.exists(firebase_audio_path) and os.path.exists(firebase_text_path):
        logger.info(f"🎤 Using cached custom voice: {character_name}")
        return firebase_audio_path, firebase_text_path
    
    # Location 3: Try to download from Firebase if it's a custom voice
    try:
        # Import voice manager (only when needed)
        import sys
        sys.path.append('/app')  # Add app directory to path
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
        
        # Use user_id parameter instead of environment variable
        if not user_id:
            logger.warning(f"⚠️ No user_id provided for custom voice: {character_name}")
            # Fallback to original paths for normal error handling
            return normal_audio_path, normal_text_path
        
        # Try to download the custom voice using the provided user_id
        logger.info(f"🔽 Attempting to download custom voice from Firebase: {character_name} for user: {user_id}")
        voice_info = voice_manager.download_custom_voice(user_id, character_name)
        
        if voice_info and os.path.exists(voice_info['audio_path']) and os.path.exists(voice_info['text_path']):
            logger.info(f"✅ Successfully downloaded custom voice: {character_name}")
            return voice_info['audio_path'], voice_info['text_path']
        else:
            logger.warning(f"⚠️ Failed to download custom voice: {character_name}")
            
    except Exception as e:
        logger.warning(f"⚠️ Error downloading custom voice {character_name}: {str(e)}")
    
    # Fallback: Return original paths (will cause normal error handling)
    # This preserves ALL existing error messages and behavior
    logger.info(f"🎤 Falling back to regular voice path for: {character_name}")
    return normal_audio_path, normal_text_path


def normalize_chinese_punctuation(text):
    """
    Convert Chinese (full-width) punctuation marks to English (half-width) equivalents.
    """
    # Mapping of Chinese punctuation to English punctuation
    chinese_to_english_punct = {
        "，": ", ",  # comma
        "。": ".",  # period
        "：": ":",  # colon
        "；": ";",  # semicolon
        "？": "?",  # question mark
        "！": "!",  # exclamation mark
        "（": "(",  # left parenthesis
        "）": ")",  # right parenthesis
        "【": "[",  # left square bracket
        "】": "]",  # right square bracket
        "《": "<",  # left angle quote
        "》": ">",  # right angle quote
        "“": '"',  # left double quotation
        "”": '"',  # right double quotation
        "‘": "'",  # left single quotation
        "’": "'",  # right single quotation
        "、": ",",  # enumeration comma
        "—": "-",  # em dash
        "…": "...",  # ellipsis
        "·": ".",  # middle dot
        "「": '"',  # left corner bracket
        "」": '"',  # right corner bracket
        "『": '"',  # left double corner bracket
        "』": '"',  # right double corner bracket
    }

    # Replace each Chinese punctuation with its English counterpart
    for zh_punct, en_punct in chinese_to_english_punct.items():
        text = text.replace(zh_punct, en_punct)

    return text


def prepare_chunk_text(
    text, chunk_method: Optional[str] = None, chunk_max_word_num: int = 100, chunk_max_num_turns: int = 1
):
    """Chunk the text into smaller pieces. We will later feed the chunks one by one to the model.

    Parameters
    ----------
    text : str
        The text to be chunked.
    chunk_method : str, optional
        The method to use for chunking. Options are "speaker", "word", "sentence", "semantic", "adaptive", "clause", or None. 
        For professional SaaS applications, "semantic" or "adaptive" are recommended for best quality.
    replace_speaker_tag_with_special_tags : bool, optional
        Whether to replace speaker tags with special tokens, by default False
        If the flag is set to True, we will replace [SPEAKER0] with <|speaker_id_start|>SPEAKER0<|speaker_id_end|>
    chunk_max_word_num : int, optional
        The maximum number of words for each chunk. Used as base limit for all methods, with adaptive methods adjusting based on content complexity, by default 100
    chunk_max_num_turns : int, optional
        The maximum number of turns for each chunk when "speaker" chunking method is used,

    Returns
    -------
    List[str]
        The list of text chunks.

    """
    if chunk_method is None:
        return [text]
    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks = []
        speaker_utterance = ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                if speaker_utterance:
                    speaker_utterance += "\n" + line
                else:
                    speaker_utterance = line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged_chunks = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged_chunk = "\n".join(speaker_chunks[i : i + chunk_max_num_turns])
                merged_chunks.append(merged_chunk)
            return merged_chunks
        return speaker_chunks
    elif chunk_method == "word":
        # TODO: We may improve the logic in the future
        # For long-form generation, we will first divide the text into multiple paragraphs by splitting with "\n\n"
        # After that, we will chunk each paragraph based on word count
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for idx, paragraph in enumerate(paragraphs):
            if language == "zh":
                # For Chinese, we will chunk based on character count
                words = list(jieba.cut(paragraph, cut_all=False))
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = "".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            else:
                words = paragraph.split(" ")
                for i in range(0, len(words), chunk_max_word_num):
                    chunk = " ".join(words[i : i + chunk_max_word_num])
                    chunks.append(chunk)
            chunks[-1] += "\n\n"
        return chunks
    elif chunk_method == "sentence":
        # Basic sentence-based chunking for better prosody and reduced hallucination
        try:
            ensure_nltk_data()
            from nltk.tokenize import sent_tokenize
        except Exception as e:
            logger.warning(f"NLTK setup failed for sentence chunking: {e}")
            logger.info("Falling back to word chunking...")
            return prepare_chunk_text(text, "word", chunk_max_word_num, chunk_max_num_turns)
        
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            if language == "zh":
                # For Chinese, use jieba for sentence segmentation
                # Simple approach: split by common Chinese sentence endings
                sentences = re.split(r'[。！？]', paragraph)
                sentences = [s.strip() for s in sentences if s.strip()]
            else:
                # For English and other languages, use NLTK sentence tokenizer
                sentences = sent_tokenize(paragraph)
            
            # Combine sentences into chunks while respecting word limits
            current_chunk = ""
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                # Check if adding this sentence would exceed the word limit
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                word_count = len(test_chunk.split())
                
                if word_count <= chunk_max_word_num:
                    current_chunk = test_chunk
                else:
                    # If current chunk has content, save it and start new chunk
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
            
            # Add the last chunk if it has content
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        # Ensure we have at least one chunk
        if not chunks:
            chunks = [text]
            
        logger.info(f"Created {len(chunks)} sentence-based chunks")
        return chunks
    elif chunk_method == "semantic":
        # Advanced semantic-aware chunking for professional SaaS
        # CRITICAL: Preserves exact input order while grouping adjacent similar sentences
        try:
            ensure_nltk_data()
            from nltk.tokenize import sent_tokenize
        except Exception as e:
            logger.warning(f"NLTK setup failed for semantic chunking: {e}")
            logger.info("Falling back to sentence chunking...")
            return prepare_chunk_text(text, "sentence", chunk_max_word_num, chunk_max_num_turns)
        
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        
        # Initialize NLP models
        nlp = ensure_spacy_model() if language != "zh" else None
        sentence_model = get_sentence_transformer()
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            if language == "zh":
                sentences = re.split(r'[。！？]', paragraph)
                sentences = [s.strip() for s in sentences if s.strip()]
            else:
                sentences = sent_tokenize(paragraph)
            
            if len(sentences) <= 1:
                chunks.append(paragraph.strip())
                continue
            
            # Calculate semantic similarity matrix for adjacent sentences only
            similarity_matrix = calculate_semantic_similarity(sentences, sentence_model)
            
            # Sequential processing with order preservation guarantee
            current_chunk = ""
            current_sentences = []
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                
                # ORDER PRESERVATION: Always process sentences in sequence
                should_start_new = False
                
                if current_sentences:
                    # Check word limit
                    test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    if len(test_chunk.split()) > chunk_max_word_num:
                        should_start_new = True
                    
                    # Check semantic similarity with IMMEDIATELY PREVIOUS sentence only
                    # This ensures we only group adjacent sentences, never rearrange
                    if i > 0 and similarity_matrix[i-1][i] < 0.3:  # Low similarity threshold
                        should_start_new = True
                
                if should_start_new and current_chunk:
                    # Save current chunk and start new one
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_sentences = [sentence]
                else:
                    # Add sentence to current chunk (maintains order)
                    current_chunk = current_chunk + " " + sentence if current_chunk else sentence
                    current_sentences.append(sentence)
            
            # Add final chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        if not chunks:
            chunks = [text]
        
        # Validate order preservation for semantic chunking
        if not validate_chunk_order_preservation(text, chunks):
            logger.warning("⚠️ Order preservation validation failed - falling back to sentence chunking")
            return prepare_chunk_text(text, "sentence", chunk_max_word_num, chunk_max_num_turns)
            
        logger.info(f"Created {len(chunks)} semantic-aware chunks (order preserved)")
        return chunks
    elif chunk_method == "adaptive":
        # Adaptive chunking based on content complexity
        try:
            ensure_nltk_data()
            from nltk.tokenize import sent_tokenize
        except Exception as e:
            logger.warning(f"NLTK setup failed for adaptive chunking: {e}")
            logger.info("Falling back to sentence chunking...")
            return prepare_chunk_text(text, "sentence", chunk_max_word_num, chunk_max_num_turns)
        
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        
        nlp = ensure_spacy_model() if language != "zh" else None
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            if language == "zh":
                sentences = re.split(r'[。！？]', paragraph)
                sentences = [s.strip() for s in sentences if s.strip()]
            else:
                sentences = sent_tokenize(paragraph)
            
            current_chunk = ""
            current_complexity = 0
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                # Calculate sentence complexity
                sentence_complexity = calculate_content_complexity(sentence, nlp)
                
                # Adaptive word limit based on complexity
                adaptive_limit = chunk_max_word_num
                if sentence_complexity > 15:  # High complexity
                    adaptive_limit = int(chunk_max_word_num * 0.7)  # Smaller chunks
                elif sentence_complexity < 8:  # Low complexity
                    adaptive_limit = int(chunk_max_word_num * 1.3)  # Larger chunks
                
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                word_count = len(test_chunk.split())
                
                if word_count <= adaptive_limit and current_complexity + sentence_complexity <= 25:
                    current_chunk = test_chunk
                    current_complexity += sentence_complexity
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_complexity = sentence_complexity
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        if not chunks:
            chunks = [text]
            
        logger.info(f"Created {len(chunks)} adaptive chunks")
        return chunks
    elif chunk_method == "clause":
        # Clause-based chunking for finer prosody control
        try:
            ensure_nltk_data()
            from nltk.tokenize import sent_tokenize
        except Exception as e:
            logger.warning(f"NLTK setup failed for clause chunking: {e}")
            logger.info("Falling back to sentence chunking...")
            return prepare_chunk_text(text, "sentence", chunk_max_word_num, chunk_max_num_turns)
        
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        
        nlp = ensure_spacy_model() if language != "zh" else None
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            if language == "zh":
                sentences = re.split(r'[。！？]', paragraph)
                sentences = [s.strip() for s in sentences if s.strip()]
            else:
                sentences = sent_tokenize(paragraph)
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                # Detect clause boundaries
                clause_boundaries = detect_clause_boundaries(sentence, nlp)
                
                if not clause_boundaries or len(sentence.split()) <= chunk_max_word_num:
                    # No clause boundaries or sentence is short enough
                    chunks.append(sentence.strip())
                else:
                    # Split at clause boundaries
                    current_clause = ""
                    last_boundary = 0
                    
                    for boundary in clause_boundaries:
                        clause = sentence[last_boundary:boundary].strip()
                        if clause:
                            test_chunk = current_clause + " " + clause if current_clause else clause
                            if len(test_chunk.split()) <= chunk_max_word_num:
                                current_clause = test_chunk
                            else:
                                if current_clause:
                                    chunks.append(current_clause.strip())
                                current_clause = clause
                        last_boundary = boundary
                    
                    # Add remaining text
                    remaining = sentence[last_boundary:].strip()
                    if remaining:
                        test_chunk = current_clause + " " + remaining if current_clause else remaining
                        if len(test_chunk.split()) <= chunk_max_word_num:
                            chunks.append(test_chunk.strip())
                        else:
                            if current_clause:
                                chunks.append(current_chunk.strip())
                            chunks.append(remaining.strip())
        
        if not chunks:
            chunks = [text]
            
        logger.info(f"Created {len(chunks)} clause-based chunks")
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")


def _build_system_message_with_audio_prompt(system_message):
    contents = []

    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN) :]

    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    ret = Message(
        role="system",
        content=contents,
    )
    return ret


class HiggsAudioModelClient:
    def __init__(
        self,
        model_path,
        audio_tokenizer,
        device=None,
        device_id=None,
        max_new_tokens=2048,
        kv_cache_lengths: List[int] = [1024, 4096, 8192],  # Multiple KV cache sizes,
        use_static_kv_cache=False,
    ):
        # Use explicit device if provided, otherwise try CUDA/MPS/CPU
        if device_id is not None:
            device = f"cuda:{device_id}"
            self._device = device
        else:
            if device is not None:
                self._device = device
            else:  # We get to choose the device
                # Prefer CUDA over MPS (Apple Silicon GPU) over CPU if available
                if torch.cuda.is_available():
                    self._device = "cuda:0"
                elif torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"

        logger.info(f"Using device: {self._device}")
        if isinstance(audio_tokenizer, str):
            # For MPS, use CPU due to embedding operation limitations in quantization layers
            audio_tokenizer_device = "cpu" if self._device == "mps" else self._device
            self._audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer, device=audio_tokenizer_device)
        else:
            self._audio_tokenizer = audio_tokenizer

        self._model = HiggsAudioModel.from_pretrained(
            model_path,
            device_map=self._device,
            torch_dtype=torch.bfloat16,
        )
        self._model.eval()
        self._kv_cache_lengths = kv_cache_lengths
        self._use_static_kv_cache = use_static_kv_cache

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._config = AutoConfig.from_pretrained(model_path)
        self._max_new_tokens = max_new_tokens
        self._collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=self._config.audio_in_token_idx,
            audio_out_token_id=self._config.audio_out_token_idx,
            audio_stream_bos_id=self._config.audio_stream_bos_id,
            audio_stream_eos_id=self._config.audio_stream_eos_id,
            encode_whisper_embed=self._config.encode_whisper_embed,
            pad_token_id=self._config.pad_token_id,
            return_audio_in_tokens=self._config.encode_audio_in_tokens,
            use_delay_pattern=self._config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self._config.audio_num_codebooks,
        )
        self.kv_caches = None
        if use_static_kv_cache:
            self._init_static_kv_cache()

    def _init_static_kv_cache(self):
        cache_config = copy.deepcopy(self._model.config.text_config)
        cache_config.num_hidden_layers = self._model.config.text_config.num_hidden_layers
        if self._model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self._model.config.audio_dual_ffn_layers)
        # A list of KV caches for different lengths
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self._model.device,
                dtype=self._model.dtype,
            )
            for length in sorted(self._kv_cache_lengths)
        }
        # Capture CUDA graphs for each KV cache length
        if "cuda" in self._device:
            logger.info(f"Capturing CUDA graphs for each KV cache length")
            self._model.capture_model(self.kv_caches.values())

    def _prepare_kv_caches(self):
        for kv_cache in self.kv_caches.values():
            kv_cache.reset()

    @torch.inference_mode()
    def generate(
        self,
        messages,
        audio_ids,
        chunked_text,
        generation_chunk_buffer_size,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        ras_win_len=7,
        ras_win_max_num_repeat=2,
        seed=123,
        *args,
        **kwargs,
    ):
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None
        sr = 24000
        audio_out_ids_l = []
        generated_audio_ids = []
        generation_messages = []
        for idx, chunk_text in tqdm.tqdm(
            enumerate(chunked_text), desc="Generating audio chunks", total=len(chunked_text)
        ):
            generation_messages.append(
                Message(
                    role="user",
                    content=chunk_text,
                )
            )
            chatml_sample = ChatMLSample(messages=messages + generation_messages)
            input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self._tokenizer)
            postfix = self._tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
            )
            input_tokens.extend(postfix)

            logger.info(f"========= Chunk {idx} Input =========")
            logger.info(self._tokenizer.decode(input_tokens))
            context_audio_ids = audio_ids + generated_audio_ids

            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1)
                if context_audio_ids
                else None,
                audio_ids_start=torch.cumsum(
                    torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0
                )
                if context_audio_ids
                else None,
                audio_waveforms_concat=None,
                audio_waveforms_start=None,
                audio_sample_rate=None,
                audio_speaker_indices=None,
            )

            batch_data = self._collator([curr_sample])
            batch = asdict(batch_data)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.contiguous().to(self._device)

            if self._use_static_kv_cache:
                self._prepare_kv_caches()

            # Generate audio
            outputs = self._model.generate(
                **batch,
                max_new_tokens=self._max_new_tokens,
                use_cache=True,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                tokenizer=self._tokenizer,
                seed=seed,
            )

            step_audio_out_ids_l = []
            for ele in outputs[1]:
                audio_out_ids = ele
                if self._config.use_delay_pattern:
                    audio_out_ids = revert_delay_pattern(audio_out_ids)
                step_audio_out_ids_l.append(audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[:, 1:-1])
            audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
            audio_out_ids_l.append(audio_out_ids)
            generated_audio_ids.append(audio_out_ids)

            generation_messages.append(
                Message(
                    role="assistant",
                    content=AudioContent(audio_url=""),
                )
            )
            if generation_chunk_buffer_size is not None and len(generated_audio_ids) > generation_chunk_buffer_size:
                generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
                generation_messages = generation_messages[(-2 * generation_chunk_buffer_size) :]

        logger.info(f"========= Final Text output =========")
        logger.info(self._tokenizer.decode(outputs[0][0]))
        concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)

        # Fix MPS compatibility: detach and move to CPU before decoding
        if concat_audio_out_ids.device.type == "mps":
            concat_audio_out_ids_cpu = concat_audio_out_ids.detach().cpu()
        else:
            concat_audio_out_ids_cpu = concat_audio_out_ids

        concat_wv = self._audio_tokenizer.decode(concat_audio_out_ids_cpu.unsqueeze(0))[0, 0]
        text_result = self._tokenizer.decode(outputs[0][0])
        return concat_wv, sr, text_result


def prepare_generation_context(scene_prompt, ref_audio, ref_audio_in_system_message, audio_tokenizer, speaker_tags, user_id=None):
    """Prepare the context for generation.

    The context contains the system message, user message, assistant message, and audio prompt if any.
    """
    system_message = None
    messages = []
    audio_ids = []
    if ref_audio is not None:
        num_speakers = len(ref_audio.split(","))
        speaker_info_l = ref_audio.split(",")
        voice_profile = None
        if any([speaker_info.startswith("profile:") for speaker_info in ref_audio.split(",")]):
            ref_audio_in_system_message = True
        if ref_audio_in_system_message:
            speaker_desc = []
            for spk_id, character_name in enumerate(speaker_info_l):
                if character_name.startswith("profile:"):
                    if voice_profile is None:
                        with open(f"{CURR_DIR}/voice_prompts/profile.yaml", "r", encoding="utf-8") as f:
                            voice_profile = yaml.safe_load(f)
                    character_desc = voice_profile["profiles"][character_name[len("profile:") :].strip()]
                    speaker_desc.append(f"SPEAKER{spk_id}: {character_desc}")
                else:
                    speaker_desc.append(f"SPEAKER{spk_id}: {AUDIO_PLACEHOLDER_TOKEN}")
            if scene_prompt:
                system_message = (
                    "Generate audio following instruction."
                    "\n\n"
                    f"<|scene_desc_start|>\n{scene_prompt}\n\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>"
                )
            else:
                system_message = (
                    "Generate audio following instruction.\n\n"
                    + f"<|scene_desc_start|>\n"
                    + "\n".join(speaker_desc)
                    + "\n<|scene_desc_end|>"
                )
            system_message = _build_system_message_with_audio_prompt(system_message)
        else:
            if scene_prompt:
                system_message = Message(
                    role="system",
                    content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>",
                )
        voice_profile = None
        for spk_id, character_name in enumerate(ref_audio.split(",")):
            if not character_name.startswith("profile:"):
                prompt_audio_path, prompt_text_path = resolve_voice_paths(character_name, base_dir=CURR_DIR, user_id=user_id)
                assert os.path.exists(prompt_audio_path), (
                    f"Voice prompt audio file {prompt_audio_path} does not exist."
                )
                assert os.path.exists(prompt_text_path), f"Voice prompt text file {prompt_text_path} does not exist."
                with open(prompt_text_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
                audio_tokens = audio_tokenizer.encode(prompt_audio_path)
                audio_ids.append(audio_tokens)

                if not ref_audio_in_system_message:
                    messages.append(
                        Message(
                            role="user",
                            content=f"[SPEAKER{spk_id}] {prompt_text}" if num_speakers > 1 else prompt_text,
                        )
                    )
                    messages.append(
                        Message(
                            role="assistant",
                            content=AudioContent(
                                audio_url=prompt_audio_path,
                            ),
                        )
                    )
    else:
        if len(speaker_tags) > 1:
            # By default, we just alternate between male and female voices
            speaker_desc_l = []

            for idx, tag in enumerate(speaker_tags):
                if idx % 2 == 0:
                    speaker_desc = f"feminine"
                else:
                    speaker_desc = f"masculine"
                speaker_desc_l.append(f"{tag}: {speaker_desc}")

            speaker_desc = "\n".join(speaker_desc_l)
            scene_desc_l = []
            if scene_prompt:
                scene_desc_l.append(scene_prompt)
            scene_desc_l.append(speaker_desc)
            scene_desc = "\n\n".join(scene_desc_l)

            system_message = Message(
                role="system",
                content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>",
            )
        else:
            system_message_l = ["Generate audio following instruction."]
            if scene_prompt:
                system_message_l.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
            system_message = Message(
                role="system",
                content="\n\n".join(system_message_l),
            )
    if system_message:
        messages.insert(0, system_message)
    return messages, audio_ids


@click.command()
@click.option(
    "--model_path",
    type=str,
    default="bosonai/higgs-audio-v2-generation-3B-base",
    help="Output wav file path.",
)
@click.option(
    "--audio_tokenizer",
    type=str,
    default="bosonai/higgs-audio-v2-tokenizer",
    help="Audio tokenizer path, if not set, use the default one.",
)
@click.option(
    "--max_new_tokens",
    type=int,
    default=2048,
    help="The maximum number of new tokens to generate.",
)
@click.option(
    "--transcript",
    type=str,
    default="transcript/single_speaker/en_dl.txt",
    help="The prompt to use for generation. If not set, we will use a default prompt.",
)
@click.option(
    "--scene_prompt",
    type=str,
    default=f"{CURR_DIR}/scene_prompts/quiet_indoor.txt",
    help="The scene description prompt to use for generation. If not set, or set to `empty`, we will leave it to empty.",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="The value used to module the next token probabilities.",
)
@click.option(
    "--top_k",
    type=int,
    default=50,
    help="The number of highest probability vocabulary tokens to keep for top-k-filtering.",
)
@click.option(
    "--top_p",
    type=float,
    default=0.95,
    help="If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
)
@click.option(
    "--ras_win_len",
    type=int,
    default=7,
    help="The window length for RAS sampling. If set to 0 or a negative value, we won't use RAS sampling.",
)
@click.option(
    "--ras_win_max_num_repeat",
    type=int,
    default=2,
    help="The maximum number of times to repeat the RAS window. Only used when --ras_win_len is set.",
)
@click.option(
    "--ref_audio",
    type=str,
    default=None,
    help="The voice prompt to use for generation. If not set, we will let the model randomly pick a voice. "
    "For multi-speaker generation, you can specify the prompts as `belinda,chadwick` and we will use the voice of belinda as SPEAKER0 and the voice of chadwick as SPEAKER1.",
)
@click.option(
    "--ref_audio_in_system_message",
    is_flag=True,
    default=False,
    help="Whether to include the voice prompt description in the system message.",
    show_default=True,
)
@click.option(
    "--chunk_method",
    default=None,
    type=click.Choice([None, "speaker", "word", "sentence", "semantic", "adaptive", "clause"]),
    help="The method to use for chunking the prompt text. Options: 'speaker' (speaker-based), 'word' (word-based), 'sentence' (sentence-based), 'semantic' (semantic-aware), 'adaptive' (complexity-adaptive), 'clause' (clause-based), or None. For professional SaaS, use 'semantic' or 'adaptive'.",
)
@click.option(
    "--chunk_max_word_num",
    default=200,
    type=int,
    help="The maximum number of words for each chunk. Used by all chunking methods as a base limit, with adaptive methods adjusting based on content complexity.",
)
@click.option(
    "--chunk_max_num_turns",
    default=1,
    type=int,
    help="The maximum number of turns for each chunk when 'speaker' chunking method is used. Only used when --chunk_method is set to 'speaker'.",
)
@click.option(
    "--generation_chunk_buffer_size",
    default=None,
    type=int,
    help="The maximal number of chunks to keep in the buffer. We will always keep the reference audios, and keep `max_chunk_buffer` chunks of generated audio.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for generation.",
)
@click.option(
    "--device_id",
    type=int,
    default=None,
    help="The device to run the model on.",
)
@click.option(
    "--user_id",
    type=str,
    default=None,
    help="User ID for custom voice resolution"
)
@click.option(
    "--out_path",
    type=str,
    default="generation.wav",
)
@click.option(
    "--use_static_kv_cache",
    type=int,
    default=1,
    help="Whether to use static KV cache for faster generation. Only works when using GPU.",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "mps", "none"]),
    default="auto",
    help="Device to use: 'auto' (pick best available), 'cuda', 'mps', or 'none' (CPU only).",
)
def main(
    model_path,
    audio_tokenizer,
    max_new_tokens,
    transcript,
    scene_prompt,
    temperature,
    top_k,
    top_p,
    ras_win_len,
    ras_win_max_num_repeat,
    ref_audio,
    ref_audio_in_system_message,
    chunk_method,
    chunk_max_word_num,
    chunk_max_num_turns,
    generation_chunk_buffer_size,
    seed,
    device_id,
    out_path,
    use_static_kv_cache,
    device,
    user_id,
):
    # specifying a device_id implies CUDA
    if device_id is None:
        if device == "auto":
            if torch.cuda.is_available():
                device_id = 0
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device_id = None  # MPS doesn't use device IDs like CUDA
                device = "mps"
            else:
                device_id = None
                device = "cpu"
        elif device == "cuda":
            device_id = 0
            device = "cuda:0"
        elif device == "mps":
            device_id = None
            device = "mps"
        else:
            device_id = None
            device = "cpu"
    else:
        device = f"cuda:{device_id}"
    # For MPS, use CPU for audio tokenizer due to embedding operation limitations
    audio_tokenizer_device = "cpu" if device == "mps" else device
    audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer, device=audio_tokenizer_device)

    # Disable static KV cache on MPS since it relies on CUDA graphs
    if device == "mps" and use_static_kv_cache:
        use_static_kv_cache = False
    model_client = HiggsAudioModelClient(
        model_path=model_path,
        audio_tokenizer=audio_tokenizer,
        device=device,
        device_id=device_id,
        max_new_tokens=max_new_tokens,
        use_static_kv_cache=use_static_kv_cache,
    )

    pattern = re.compile(r"\[(SPEAKER\d+)\]")

    if os.path.exists(transcript):
        logger.info(f"Loading transcript from {transcript}")
        with open(transcript, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

    if scene_prompt is not None and scene_prompt != "empty" and os.path.exists(scene_prompt):
        with open(scene_prompt, "r", encoding="utf-8") as f:
            scene_prompt = f.read().strip()
    else:
        scene_prompt = None

    speaker_tags = sorted(set(pattern.findall(transcript)))
    # Perform some basic normalization
    transcript = normalize_chinese_punctuation(transcript)
    # Other normalizations (e.g., parentheses and other symbols. Will be improved in the future)
    transcript = transcript.replace("(", " ")
    transcript = transcript.replace(")", " ")
    transcript = transcript.replace("°F", " degrees Fahrenheit")
    transcript = transcript.replace("°C", " degrees Celsius")

    for tag, replacement in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE_s>[Humming]</SE_s>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        transcript = transcript.replace(tag, replacement)
    lines = transcript.split("\n")
    transcript = "\n".join([" ".join(line.split()) for line in lines if line.strip()])
    transcript = transcript.strip()

    if not any([transcript.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        transcript += "."

    messages, audio_ids = prepare_generation_context(
        scene_prompt=scene_prompt,
        ref_audio=ref_audio,
        ref_audio_in_system_message=ref_audio_in_system_message,
        audio_tokenizer=audio_tokenizer,
        speaker_tags=speaker_tags,
        user_id=user_id,
    )
    chunked_text = prepare_chunk_text(
        transcript,
        chunk_method=chunk_method,
        chunk_max_word_num=chunk_max_word_num,
        chunk_max_num_turns=chunk_max_num_turns,
    )

    logger.info("Chunks used for generation:")
    for idx, chunk_text in enumerate(chunked_text):
        logger.info(f"Chunk {idx}:")
        logger.info(chunk_text)
        logger.info("-----")

    concat_wv, sr, text_output = model_client.generate(
        messages=messages,
        audio_ids=audio_ids,
        chunked_text=chunked_text,
        generation_chunk_buffer_size=generation_chunk_buffer_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        ras_win_len=ras_win_len,
        ras_win_max_num_repeat=ras_win_max_num_repeat,
        seed=seed,
    )

    sf.write(out_path, concat_wv, sr)
    logger.info(f"Wav file is saved to '{out_path}' with sample rate {sr}")


if __name__ == "__main__":
    main()
