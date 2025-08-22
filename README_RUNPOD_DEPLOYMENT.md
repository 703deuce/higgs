# Higgs Audio V2 RunPod Serverless Deployment Guide

This guide provides step-by-step instructions for deploying Higgs Audio V2 as a serverless API endpoint on RunPod with GitHub integration.

## Quick Start

1. **Fork this repository** to your GitHub account
2. **Create a RunPod serverless endpoint** pointing to your fork
3. **Configure the environment** and deploy
4. **Test your endpoint** with API calls

## Prerequisites

- GitHub account
- RunPod account with sufficient credits
- GPU with at least 24GB VRAM (recommended: A40, A100, or RTX 4090/6000 Ada)
- Basic knowledge of REST APIs

## Step-by-Step Deployment

### 1. Fork the Repository

1. Go to [boson-ai/higgs-audio](https://github.com/boson-ai/higgs-audio)
2. Click "Fork" to create your own copy
3. Clone your fork locally (optional, for customization):

```bash
git clone https://github.com/YOUR_USERNAME/higgs-audio.git
cd higgs-audio
```

### 2. Add Serverless Files

The following files have been added for serverless deployment:

- `handler.py` - Basic serverless handler
- `handler_optimized.py` - Optimized version with cold start improvements
- `requirements-serverless.txt` - Dependencies for serverless deployment
- `Dockerfile` - Container configuration
- `README_RUNPOD_DEPLOYMENT.md` - This deployment guide

If you cloned the repository, commit these files:

```bash
git add handler.py handler_optimized.py requirements-serverless.txt Dockerfile README_RUNPOD_DEPLOYMENT.md
git commit -m "Add RunPod serverless deployment files"
git push origin main
```

### 3. Create RunPod Serverless Endpoint

1. **Login to RunPod**
   - Go to [RunPod](https://runpod.io/) and sign in
   - Navigate to "Serverless" section

2. **Create New Endpoint**
   - Click "Create Endpoint"
   - Choose "GitHub" as the source

3. **Configure GitHub Integration**
   - Connect your GitHub account
   - Select your forked `higgs-audio` repository
   - Set branch to `main`

4. **Configure Container Settings**
   - **Docker Image**: Use the default or specify custom base image
   - **Entry Point**: `handler.py` (or `handler_optimized.py` for better performance)
   - **Handler Function**: `handler`

5. **Hardware Configuration**
   - **GPU**: Select GPU with ≥24GB VRAM
     - Recommended: RTX A6000 (48GB), A40 (48GB), A100 (40GB/80GB)
     - Minimum: RTX 4090 (24GB), RTX 6000 Ada (48GB)
   - **CPU**: 8+ cores recommended
   - **RAM**: 32GB+ recommended
   - **Storage**: 50GB+ for model caching

6. **Environment Variables** (Optional)
   ```
   MODEL_PATH=bosonai/higgs-audio-v2-generation-3B-base
   AUDIO_TOKENIZER_PATH=bosonai/higgs-audio-v2-tokenizer
   DEVICE=cuda
   ENABLE_TORCH_COMPILE=true
   MAX_TEXT_LENGTH=10000
   PRELOAD_MODELS=true
   ```

7. **Scaling Configuration**
   - **Max Workers**: 1-3 (depending on your needs)
   - **Idle Timeout**: 30 seconds
   - **Request Timeout**: 120 seconds

### 4. Deploy and Test

1. **Deploy**
   - Click "Deploy" 
   - Wait for the build process (10-15 minutes first time)
   - Monitor logs for any errors

2. **Get Endpoint URL**
   - Once deployed, copy your endpoint URL
   - It will look like: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run`

3. **Test the Endpoint**

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "text": "Hello, this is a test of Higgs Audio running on RunPod!",
      "temperature": 0.7,
      "max_new_tokens": 512,
      "output_format": "wav"
    }
  }'
```

## API Reference

### Request Format

```json
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
    "ras_win_len": 7,
    "ras_win_max_num_repeat": 2
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | **required** | Text to convert to speech |
| `max_new_tokens` | integer | 512 | Maximum tokens to generate |
| `temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `top_k` | integer | 50 | Top-k sampling parameter |
| `top_p` | float | 0.95 | Top-p sampling parameter |
| `seed` | integer | null | Random seed for reproducibility |
| `force_audio_gen` | boolean | true | Force audio generation |
| `system_prompt` | string | "Generate audio..." | System prompt for context |
| `output_format` | string | "wav" | Output format: "wav", "mp3", "base64" |
| `ref_audio_base64` | string | null | Reference audio for voice cloning |
| `scene_description` | string | null | Scene description for context |
| `ras_win_len` | integer | 7 | RAS window length |
| `ras_win_max_num_repeat` | integer | 2 | RAS max repetitions |

### Response Format

```json
{
  "audio_base64": "UklGRkK0AQBXQV...",
  "sampling_rate": 24000,
  "duration": 5.2,
  "format": "wav",
  "content_type": "audio/wav",
  "generated_text": "Generated text response",
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 200,
    "total_tokens": 250
  },
  "timing": {
    "total_time": 8.5,
    "generation_time": 7.2,
    "audio_duration": 5.2
  },
  "model_info": {
    "model_path": "bosonai/higgs-audio-v2-generation-3B-base",
    "device": "cuda",
    "warmup_completed": true
  }
}
```

## Advanced Features

### Voice Cloning with Reference Audio

Include a base64-encoded reference audio file:

```json
{
  "input": {
    "text": "Clone this voice and say hello!",
    "ref_audio_base64": "UklGRkK0AQBXQV...",
    "temperature": 0.3
  }
}
```

### Multi-Speaker Dialog

Use speaker tags in your text:

```json
{
  "input": {
    "text": "[SPEAKER0] Hello there! [SPEAKER1] Hi, how are you today?",
    "system_prompt": "Generate multi-speaker dialog audio.",
    "scene_description": "Two friends meeting at a coffee shop"
  }
}
```

### Scene Context

Add environmental context:

```json
{
  "input": {
    "text": "Welcome to our quiet library.",
    "scene_description": "Indoor library setting with soft ambient sounds"
  }
}
```

## Performance Optimization

### Using the Optimized Handler

For better performance, use `handler_optimized.py`:

1. Change the entry point in RunPod to `handler_optimized.py`
2. Set environment variables:
   ```
   ENABLE_TORCH_COMPILE=true
   PRELOAD_MODELS=true
   ```

### Cold Start Optimization

- Models are pre-downloaded during container startup
- First request includes automatic model warmup
- Static KV caches are pre-allocated for common sequence lengths
- PyTorch settings are optimized for inference

### Memory Management

- Automatic GPU memory fraction allocation (90%)
- Empty cache management between requests
- Optimized batch processing

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Use smaller `max_new_tokens` values
   - Reduce KV cache sizes in environment variables
   - Choose GPU with more VRAM

2. **Long Cold Start Times**
   - Use the optimized handler (`handler_optimized.py`)
   - Enable model pre-loading with `PRELOAD_MODELS=true`
   - Consider using a persistent endpoint instead

3. **Audio Quality Issues**
   - Adjust `temperature` (lower = more deterministic)
   - Try different `ras_win_len` values
   - Check input text quality and formatting

4. **Request Timeouts**
   - Reduce `max_new_tokens`
   - Increase request timeout in RunPod settings
   - Use faster GPU if available

### Monitoring and Logs

- Check RunPod logs for detailed error information
- Monitor GPU utilization and memory usage
- Track request timing metrics in response

### Getting Help

- **Higgs Audio Issues**: [GitHub Issues](https://github.com/boson-ai/higgs-audio/issues)
- **RunPod Support**: [RunPod Discord](https://discord.gg/runpod)
- **API Documentation**: Check the serve engine source code for advanced parameters

## Pricing Considerations

- **GPU Costs**: Vary by GPU type and region
- **Idle Time**: Configure appropriate idle timeout
- **Storage**: Consider model caching costs
- **Network**: Audio file transfer costs

### Cost Optimization Tips

1. Use appropriate GPU size for your use case
2. Set reasonable idle timeouts
3. Batch requests when possible
4. Monitor usage patterns and adjust scaling

## Security Considerations

- **API Keys**: Keep your RunPod API key secure
- **Input Validation**: The handler includes input validation
- **Rate Limiting**: Consider implementing rate limiting for production
- **Audio Content**: Be aware of audio content policies

## Production Deployment

For production use, consider:

1. **Multiple Endpoints**: Deploy across different regions
2. **Load Balancing**: Implement client-side load balancing
3. **Monitoring**: Set up comprehensive monitoring
4. **Backup**: Have fallback endpoints ready
5. **Caching**: Implement response caching for repeated requests

## License

This deployment guide is provided under the same license as the Higgs Audio project (Apache-2.0).

## Contributing

Feel free to submit improvements to the deployment configuration:

1. Fork the repository
2. Make your changes
3. Submit a pull request

---

For additional support, please refer to the main [Higgs Audio repository](https://github.com/boson-ai/higgs-audio) and [RunPod documentation](https://docs.runpod.ai/).
