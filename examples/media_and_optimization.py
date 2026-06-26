"""Example: ingest images / audio / video and tune training speed (v1.0.1).

doc2lora can pull text out of more than just documents:

  - Images (OCR):   pip install "doc2lora[image]"   # + the tesseract-ocr binary
  - Audio (Whisper): pip install "doc2lora[audio]"   # + the ffmpeg binary
  - Video (both):   pip install "doc2lora[video]"   # opencv + tesseract + whisper

Audio/video transcription defaults to faster-whisper and falls back to
openai-whisper, then SpeechRecognition. Image and video OCR use pytesseract.

The training-speed knobs below are split into "applied automatically" (TF32,
bf16/fp16, fused AdamW, SDPA, pinned-memory gating - all hardware-aware and safe)
and "opt-in" (shown here). See the README "Training speed optimizations" section.
"""

import sys
from pathlib import Path

# import doc2lora from the repo without installing it
sys.path.insert(0, str(Path(__file__).parent.parent))

from doc2lora import convert  # noqa: E402


def main():
    docs_path = Path(__file__).parent / "example_documents"

    adapter_path = convert(
        documents_path=str(docs_path),
        output_path="media_adapter.json",
        model_name="microsoft/DialoGPT-small",
        num_epochs=3,
        # ---- media ingestion ----
        audio_backend="faster-whisper",  # or openai-whisper / speech_recognition
        whisper_model_size="base",  # tiny..large-v3
        ocr_languages="eng",  # tesseract language(s), e.g. "eng+fra"
        video_frame_interval=1.0,  # OCR one frame per second of video
        max_workers=8,  # parse the folder with 8 threads
        # ---- opt-in training speedups ----
        group_by_length=True,  # cut padding on variable-length corpora
        # torch_compile defaults to None -> auto-on for CUDA + corpora >=10MB text;
        # pass torch_compile=False to force it off, True to always compile
        # attn_implementation="flash_attention_2",  # Ampere+ CUDA + flash-attn
        # dataloader_num_workers=4,      # parallel data loading (Linux/CUDA)
    )

    print(f"✅ Adapter written to: {adapter_path}")
    print(
        "\nMixed media (images, audio, video) in the folder is turned into text "
        "automatically - run `doc2lora formats` to see every supported extension."
    )


if __name__ == "__main__":
    main()
