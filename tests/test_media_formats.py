"""Tests for the image / video / whisper formats and parallel parsing (v1.0.1)."""

import types

import pytest

import doc2lora.parsers as parsers_mod
from doc2lora.parsers import DocumentParser


@pytest.fixture
def parser():
    return DocumentParser()


# ---- extension registration ----


def test_image_and_video_extensions_registered(parser):
    for ext in (".png", ".jpg", ".bmp", ".gif", ".tiff", ".webp", ".svg"):
        assert ext in parser.IMAGE_EXTENSIONS
        assert ext in parser.DOCUMENT_EXTENSIONS
    for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        assert ext in parser.VIDEO_EXTENSIONS
        assert ext in parser.DOCUMENT_EXTENSIONS
    assert ".svg" in parser.VECTOR_IMAGE_EXTENSIONS
    assert ".png" in parser.RASTER_IMAGE_EXTENSIONS


def test_default_audio_backend_is_faster_whisper(parser):
    assert parser.audio_backend == "faster-whisper"


# ---- svg (vector text, no optional dep) ----


def test_parse_svg_extracts_markup_text(parser, temp_dir):
    svg = temp_dir / "diagram.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><title>Chart</title>'
        "<text>Quarterly revenue</text><text>Up 20%</text></svg>"
    )

    result = parser.parse_file(svg)

    assert result is not None
    assert result["extension"] == ".svg"
    assert "Quarterly revenue" in result["content"]
    assert "Up 20%" in result["content"]
    assert "Chart" in result["content"]


# ---- raster image OCR (pytesseract mocked, real Pillow) ----


def test_parse_image_ocr(temp_dir, monkeypatch):
    pytest.importorskip("PIL")
    from PIL import Image

    img = temp_dir / "scan.png"
    Image.new("RGB", (32, 12), "white").save(img)

    # patch in real Pillow + a fake tesseract so no system binary is needed
    monkeypatch.setattr(parsers_mod, "Image", Image)
    monkeypatch.setattr(
        parsers_mod,
        "pytesseract",
        types.SimpleNamespace(image_to_string=lambda *a, **k: "  recognized words  "),
    )

    result = DocumentParser().parse_file(img)

    assert result is not None
    assert result["extension"] == ".png"
    assert result["content"] == "recognized words"


def test_parse_image_without_deps_returns_none(temp_dir, monkeypatch):
    """No OCR backend -> empty content -> file skipped, not a crash."""
    pytest.importorskip("PIL")
    from PIL import Image

    img = temp_dir / "scan.png"
    Image.new("RGB", (8, 8), "white").save(img)

    monkeypatch.setattr(parsers_mod, "pytesseract", None)

    result = DocumentParser().parse_file(img)

    assert result is None


# ---- video (audio transcript + per-frame OCR, deduped) ----


class _FakeCapture:
    """Minimal cv2.VideoCapture stand-in yielding scripted frames."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def isOpened(self):
        return True

    def get(self, prop):
        return 1.0  # 1 fps -> sample every frame

    def read(self):
        if self._i < len(self._frames):
            frame = self._frames[self._i]
            self._i += 1
            return True, frame
        return False, None

    def release(self):
        pass


def test_parse_video_combines_transcript_and_deduped_ocr(temp_dir, monkeypatch):
    # a slide shown across 2 frames, a new slide, then the first slide again
    ocr_script = iter(["Slide A", "Slide A", "Slide B", "Slide A"])

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=lambda path: _FakeCapture([0, 1, 2, 3]),
        CAP_PROP_FPS=0,
        COLOR_BGR2GRAY=6,
        cvtColor=lambda frame, code: frame,
    )
    monkeypatch.setattr(parsers_mod, "cv2", fake_cv2)
    monkeypatch.setattr(
        parsers_mod,
        "pytesseract",
        types.SimpleNamespace(image_to_string=lambda *a, **k: next(ocr_script)),
    )

    parser = DocumentParser()
    # stub transcription so the test doesn't touch a whisper model
    monkeypatch.setattr(parser, "_transcribe_media", lambda fp: "spoken narration")

    vid = temp_dir / "demo.mp4"
    vid.write_bytes(b"not-a-real-mp4")

    result = parser.parse_file(vid)

    assert result is not None
    assert result["extension"] == ".mp4"
    content = result["content"]
    assert "spoken narration" in content
    assert "Slide A" in content and "Slide B" in content
    # identical on-screen text is deduped: "Slide A" appears once despite 3 frames
    assert content.count("Slide A") == 1


# ---- audio/video transcription backend resolution ----


def test_resolve_audio_backend_prefers_faster_whisper(monkeypatch):
    monkeypatch.setattr(parsers_mod, "WhisperModel", object())
    monkeypatch.setattr(parsers_mod, "openai_whisper", object())
    monkeypatch.setattr(parsers_mod, "sr", object())
    assert DocumentParser()._resolve_audio_backend() == "faster-whisper"


def test_resolve_audio_backend_falls_back(monkeypatch):
    # requested backend missing -> next available in preference order
    monkeypatch.setattr(parsers_mod, "WhisperModel", None)
    monkeypatch.setattr(parsers_mod, "openai_whisper", None)
    monkeypatch.setattr(parsers_mod, "sr", object())
    parser = DocumentParser(audio_backend="faster-whisper")
    assert parser._resolve_audio_backend() == "speech_recognition"


def test_resolve_audio_backend_none_available(monkeypatch):
    monkeypatch.setattr(parsers_mod, "WhisperModel", None)
    monkeypatch.setattr(parsers_mod, "openai_whisper", None)
    monkeypatch.setattr(parsers_mod, "sr", None)
    assert DocumentParser()._resolve_audio_backend() is None


# ---- parallel directory parsing ----


def test_parse_directory_parallel_stable_order(parser, temp_dir):
    (temp_dir / "sub").mkdir()
    (temp_dir / "a.md").write_text("# A")
    (temp_dir / "b.txt").write_text("bee")
    (temp_dir / "sub" / "c.json").write_text('{"x": 1}')

    docs = parser.parse_directory(str(temp_dir), max_workers=4)

    names = [d["filename"] for d in docs]
    assert set(names) == {"a.md", "b.txt", "c.json"}
    # returned order is stable (sorted by path) regardless of completion order
    paths = [d["filepath"] for d in docs]
    assert paths == sorted(paths)


def test_resolve_workers_auto_and_explicit():
    assert DocumentParser._resolve_workers(3, 20) == 3
    assert DocumentParser._resolve_workers(None, 1) == 1
    assert DocumentParser._resolve_workers(None, 50) >= 1
    assert DocumentParser._resolve_workers(0, 50) == 1
