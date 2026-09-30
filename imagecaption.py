#WEEK 2
"""Image captioning with BLIP-1 (Hugging Face)."""
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_NAME = "Salesforce/blip-image-captioning-base"

_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _model is None:
        _processor = BlipProcessor.from_pretrained(MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
        _model.eval()
    return _processor, _model


def generate_caption(image: Image.Image) -> str:
    """Takes a PIL image and returns a text caption."""
    processor, model = _load_model()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(output_ids[0], skip_special_tokens=True)
