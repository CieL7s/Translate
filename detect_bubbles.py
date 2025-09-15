import torch.serialization
from ultralytics import YOLO
import os

# Global cache untuk model
_cached_model = None
_cached_model_path = None

def detect_bubbles(model_path, image_path):
    """
    Detects bubbles in an image using cached YOLO model with fallback.
    """
    global _cached_model, _cached_model_path
    
    # Load model cuma sekali
    if _cached_model is None or _cached_model_path != model_path:
        
        # Try load model yang diminta
        try:
            with torch.serialization.safe_globals([YOLO]):
                _cached_model = YOLO(model_path)
            _cached_model_path = model_path
            print(f"✅ Model loaded: {model_path}")
        except:
            # Fallback ke model.pt
            print(f"❌ {model_path} failed, trying model.pt...")
            try:
                with torch.serialization.safe_globals([YOLO]):
                    _cached_model = YOLO("model.pt")
                _cached_model_path = "model.pt"
                print("✅ Fallback to model.pt successful")
            except:
                print("❌ All models failed!")
                return []

    # Run inference dengan settings optimal
    try:
        # ONNX = lebih agresif, PyTorch = lebih konservatif
        if _cached_model_path.endswith('.onnx'):
            results = _cached_model(image_path, imgsz=320, conf=0.25, verbose=False)[0]
        else:
            results = _cached_model(image_path, imgsz=416, conf=0.3, verbose=False)[0]
        
        return results.boxes.data.tolist()
    except:
        return []

# Alias untuk compatibility
def detect_bubbles_cached(model_path, image_path):
    return detect_bubbles(model_path, image_path)

def detect_bubbles_quantized(model_path, image_path):
    return detect_bubbles(model_path, image_path)