# Firebase Storage Path Feature

## Overview

The endpoint now supports downloading custom voices directly from Firebase Storage using a path or URL, eliminating the need for pre-installed voices on the worker.

## New API Parameters

### `ref_audio_firebase_path` (string, optional)
- **Description**: Firebase Storage path or full download URL for the audio file
- **Examples**:
  - Firebase Storage path: `user_voices/user123/voice_name.wav`
  - Full URL: `https://firebasestorage.googleapis.com/v0/b/bucket/o/path%2Fto%2Ffile.wav?alt=media`

### `ref_audio_firebase_text_path` (string, optional)
- **Description**: Firebase Storage path or full download URL for the text transcription file
- **Note**: If not provided, the system will automatically try to infer the text path from the audio path (e.g., `voice.wav` → `voice.txt`)

## Usage Examples

### Example 1: Using Firebase Storage Path

```json
{
  "input": {
    "text": "Hello, this is a test of the voice cloning feature.",
    "ref_audio_firebase_path": "user_voices/krxbH4KpoYTWaZKIKIDIzkJx62/cloned_1756205093378_bzbfy4n4x_Maya_Pop_Culture_Queen.wav",
    "temperature": 0.3,
    "seed": 12345
  }
}
```

### Example 2: Using Full Firebase Download URL

```json
{
  "input": {
    "text": "Hello, this is a test of the voice cloning feature.",
    "ref_audio_firebase_path": "https://firebasestorage.googleapis.com/v0/b/aitts-d4c6d.firebasestorage.app/o/user_voices%2Fuser123%2Fvoice.wav?alt=media",
    "ref_audio_firebase_text_path": "https://firebasestorage.googleapis.com/v0/b/aitts-d4c6d.firebasestorage.app/o/user_voices%2Fuser123%2Fvoice.txt?alt=media",
    "temperature": 0.3
  }
}
```

### Example 3: With Explicit Text Path

```json
{
  "input": {
    "text": "Hello, this is a test of the voice cloning feature.",
    "ref_audio_firebase_path": "user_voices/user123/voice.wav",
    "ref_audio_firebase_text_path": "user_voices/user123/voice.txt",
    "temperature": 0.3
  }
}
```

## How It Works

1. **Download**: The handler downloads the audio and text files from Firebase Storage
2. **Cache**: Files are temporarily cached in `/runpod-volume/temp_voices/`
3. **Prepare**: Files are copied to `/app/examples/voice_prompts/` for generation.py to access
4. **Generate**: The voice is used for text-to-speech generation
5. **Cleanup**: Temporary files are cleaned up after generation

## Benefits

✅ **No Pre-installation Required**: Voices don't need to be pre-installed on the worker  
✅ **Flexible**: Works with any Firebase Storage path or URL  
✅ **Frontend-Friendly**: Frontend can directly pass Firebase paths without knowing worker structure  
✅ **Automatic Text Inference**: Text file path is automatically inferred if not provided  
✅ **Backward Compatible**: Existing `ref_audio_name` and `ref_audio_base64` methods still work  

## Comparison with Other Methods

| Method | Use Case | Requires Pre-installation |
|--------|----------|---------------------------|
| `ref_audio_name` | Built-in or pre-downloaded voices | ✅ Yes (or requires user_id for custom) |
| `ref_audio_base64` | Direct audio data upload | ❌ No |
| `ref_audio_firebase_path` | Firebase Storage voices | ❌ No |

## Error Handling

- If the Firebase path is invalid or file doesn't exist, the request will fail with a clear error message
- If the text file is missing, an empty text file will be created as a fallback
- Network timeouts are set to 30 seconds per file download

## Notes

- Only one reference audio method can be used at a time (`ref_audio_name`, `ref_audio_firebase_path`, or `ref_audio_base64`)
- The voice files are downloaded on-demand for each request
- Files are cached temporarily but cleaned up after generation
- For production use, consider implementing a more persistent cache strategy

