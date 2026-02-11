import os
import sys
import tempfile
from typing import Iterator

import numpy as np
import soundfile as sf
from cog import BasePredictor, Input, Path

# Set working directory to GPT-SoVITS code
GPT_SOVITS_DIR = "/src/GPT-SoVITS"
GPT_SOVITS_CODE = "/src/GPT-SoVITS/GPT_SoVITS"
os.chdir(GPT_SOVITS_DIR)
sys.path.insert(0, GPT_SOVITS_DIR)
sys.path.insert(0, GPT_SOVITS_CODE)

REF_AUDIOS = {
    "happy": {
        "path": "/src/ref_audio/belle_ref.wav",
        "text": "交给我吧！不管是传奇绳匠的名声，还是市长特使的声誉，我都不会辜负的。",
    },
    "calm": {
        "path": "/src/ref_audio/belle_ref_calm.wav",
        "text": "但愿如此，总之政务的调查也好，找人也好，现在我们能做的都只有等消息了。",
    },
}

CONFIG_PATH = "/src/config/firefly.yaml"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory."""
        from TTS_infer_pack.TTS import TTS, TTS_Config

        config = TTS_Config(CONFIG_PATH)
        config.device = "cuda"
        config.is_half = True
        self.tts = TTS(config)

    def predict(
        self,
        text: str = Input(description="Text to synthesize"),
        text_lang: str = Input(
            description="Language of the text",
            default="zh",
            choices=["zh", "en", "ja", "ko", "yue", "auto", "auto_yue"],
        ),
        ref_style: str = Input(
            description="Reference audio style",
            default="happy",
            choices=["happy", "calm"],
        ),
        speed_factor: float = Input(
            description="Speed factor", default=0.95, ge=0.5, le=2.0
        ),
        top_k: int = Input(description="Top-k sampling", default=15, ge=1, le=100),
        top_p: float = Input(
            description="Top-p sampling", default=0.95, ge=0.0, le=1.0
        ),
        temperature: float = Input(
            description="Temperature", default=0.95, ge=0.01, le=2.0
        ),
    ) -> Path:
        """Run TTS inference."""
        ref = REF_AUDIOS[ref_style]

        inputs = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": ref["path"],
            "prompt_text": ref["text"],
            "prompt_lang": "zh",
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "speed_factor": speed_factor,
            "text_split_method": "cut4",
            "batch_size": 1,
            "fragment_interval": 0.3,
            "seed": -1,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
        }

        audio_chunks = []
        sample_rate = None
        for sr, chunk in self.tts.run(inputs):
            sample_rate = sr
            audio_chunks.append(chunk)

        if not audio_chunks:
            raise RuntimeError("TTS produced no audio output")

        audio = np.concatenate(audio_chunks)
        output_path = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(str(output_path), audio, sample_rate)
        return output_path
