# GPT-SoVITS Belle TTS (Replicate)

Deploy GPT-SoVITS belle_combined model to Replicate.

## Setup

1. Copy GPT-SoVITS code + weights to `GPT-SoVITS/`
2. Copy ref audio to `ref_audio/` (belle_ref.wav, belle_ref_calm.wav)
3. Copy config to `config/` (firefly.yaml)

## Deploy

```bash
export PATH=$HOME/.local/bin:$PATH
export REPLICATE_API_TOKEN=your_token
cog push r8.im/xslingcn/gpt-sovits-belle
```

## Structure

- `cog.yaml` - Environment definition
- `predict.py` - Inference interface
- `GPT-SoVITS/` - Code + model weights (not in git)
- `ref_audio/` - Reference audio files (not in git)
- `config/` - Model config (not in git)
