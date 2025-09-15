from add_text import add_text
from detect_bubbles import detect_bubbles
from process_bubble import process_bubble
from translator import MangaTranslator
from ultralytics import YOLO
from manga_ocr import MangaOcr
from PIL import Image
import gradio as gr
import numpy as np
import cv2
import torch
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Optimize PyTorch untuk ThinkPad X250
torch.set_num_threads(4)
torch.set_grad_enabled(False)

# Pilih model terbaik yang available
if os.path.exists("model.onnx"):
    MODEL = "model.onnx"
    print("✅ Using ONNX model (fastest)")
elif os.path.exists("model.torchscript"):
    MODEL = "model.torchscript" 
    print("✅ Using TorchScript model")
else:
    MODEL = "model.pt"
    print("✅ Using original PyTorch model")

EXAMPLE_LIST = [["examples/0.png"], ["examples/ex0.png"]]
TITLE = "Manga Translator (Parallel Optimized)"
DESCRIPTION = f"Translate manga bubbles with parallel processing! Using {MODEL} for better performance."

# Load translator & OCR sekali aja
print("🚀 Loading models...")
manga_translator = MangaTranslator()
mocr = MangaOcr()
print("✅ Ready!")

def process_single_bubble(bubble_data):
    """Process a single bubble: OCR + Translation + Text placement"""
    try:
        image, result, font, translation_method = bubble_data
        x1, y1, x2, y2, score, class_id = result
        
        # Skip confidence rendah
        if score < 0.25:
            return None
            
        detected_image = image[int(y1):int(y2), int(x1):int(x2)]
        
        # Skip bubble terlalu kecil
        if detected_image.shape[0] < 30 or detected_image.shape[1] < 30:
            return None
            
        # OCR
        im = Image.fromarray(np.uint8((detected_image)*255))
        text = mocr(im)
        
        # Skip kalau ga ada text
        if not text or len(text.strip()) < 2:
            return None
            
        # Process bubble dan translate secara parallel
        processed_image, cont = process_bubble(detected_image.copy())
        text_translated = manga_translator.translate(text, method=translation_method)
        
        # Add text
        final_image = add_text(processed_image, text_translated, font, cont)
        
        return {
            'coords': (int(y1), int(y2), int(x1), int(x2)),
            'image': final_image,
            'original_text': text,
            'translated_text': text_translated
        }
        
    except Exception as e:
        print(f"Error processing bubble: {e}")
        return None

def process_bubbles_batch(bubble_batch):
    """Process a batch of bubbles in parallel"""
    results = []
    
    # Use ThreadPoolExecutor untuk I/O bound operations (translation API calls)
    with ThreadPoolExecutor(max_workers=3) as executor:  # Limit untuk API rate limits
        future_to_bubble = {executor.submit(process_single_bubble, bubble): i 
                           for i, bubble in enumerate(bubble_batch)}
        
        for future in as_completed(future_to_bubble):
            result = future.result()
            if result is not None:
                results.append(result)
    
    return results

def predict(img, translation_method, font):
    start_time = time.time()
    
    if translation_method == None:
        translation_method = "hf"  # Fastest offline
    if font == None:
        font = "fonts/animeace_i.ttf"

    # Detect bubbles
    print("🔍 Detecting bubbles...")
    results = detect_bubbles(MODEL, img)
    
    image = np.array(img)
    
    if not results:
        print("❌ No bubbles detected")
        return Image.fromarray(image)
    
    # Prepare bubble data for parallel processing
    bubble_data = []
    for result in results[:15]:  # Process max 15 bubbles
        bubble_data.append((image, result, font, translation_method))
    
    print(f"📝 Processing {len(bubble_data)} bubbles in parallel...")
    
    # Process bubbles in batches to manage memory and API limits
    batch_size = 5  # Process 5 bubbles at a time
    all_results = []
    
    for i in range(0, len(bubble_data), batch_size):
        batch = bubble_data[i:i + batch_size]
        batch_results = process_bubbles_batch(batch)
        all_results.extend(batch_results)
        
        # Small delay between batches to respect API limits
        if i + batch_size < len(bubble_data):
            time.sleep(0.5)
    
    # Apply results to image
    print(f"✍️ Applying {len(all_results)} processed bubbles to image...")
    for result in all_results:
        y1, y2, x1, x2 = result['coords']
        image[y1:y2, x1:x2] = result['image']
        print(f"   '{result['original_text'][:20]}...' -> '{result['translated_text'][:20]}...'")
    
    processing_time = time.time() - start_time
    print(f"⚡ Total processing time: {processing_time:.1f}s")
    
    return Image.fromarray(image)

demo = gr.Interface(
    fn=predict,
    inputs=[
        "image",
        gr.Dropdown([
            ("Helsinki-NLP (Fast, Offline)", "hf"),
            ("Google", "google"),
            ("Sogou", "sogou"),
            ("Bing", "bing"),
            ("Double Translation", "db")
        ],
        label="Translation Method",
        value="hf"),  # Default to fastest offline method
        gr.Dropdown([
            ("animeace_i", "fonts/animeace_i.ttf"),
            ("mangati", "fonts/mangati.ttf"),
            ("ariali", "fonts/ariali.ttf")
        ],
        label="Text Font",
        value="fonts/animeace_i.ttf")
    ],
    outputs=[gr.Image()],
    examples=EXAMPLE_LIST,
    title=TITLE,
    description=DESCRIPTION
)

if __name__ == "__main__":
    print(f"🎯 Using {MODEL} with parallel processing")
    print("⚡ Expected speedup: 2-4x faster than sequential processing")
    demo.launch(debug=False, share=False)