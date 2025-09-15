"""
Universal YOLO model optimizer - works with any .pt file
Usage: python onix.py <model_path> [options]
Examples:
  python onix.py yolov8n.pt
  python onix.py yolov11s.pt --imgsz 416
  python onix.py custom_model.pt --formats onnx,torchscript
"""
import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

def get_model_info(model_path):
    """Get basic info about the model"""
    try:
        model = YOLO(model_path)
        return {
            'path': model_path,
            'size_mb': os.path.getsize(model_path) / (1024*1024),
            'task': getattr(model, 'task', 'detect'),
            'loaded': True
        }
    except Exception as e:
        return {
            'path': model_path,
            'size_mb': 0,
            'error': str(e),
            'loaded': False
        }

def create_optimized_models(model_path, output_dir=None, formats=None, imgsz=320):
    """
    Create optimized versions of any YOLO model for faster inference
    """
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"📦 Loading model: {model_path}")
    
    # Get model info
    info = get_model_info(model_path)
    if not info['loaded']:
        print(f"❌ Failed to load model: {info.get('error', 'Unknown error')}")
        return False
    
    print(f"   📊 Size: {info['size_mb']:.1f} MB")
    print(f"   🎯 Task: {info.get('task', 'detect')}")
    
    model = YOLO(model_path)
    
    # Determine output directory and base name
    model_path_obj = Path(model_path)
    base_name = model_path_obj.stem  # filename without extension
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    else:
        output_dir = model_path_obj.parent
    
    print(f"📁 Output directory: {output_dir}")
    
    # Default formats if not specified
    if formats is None:
        formats = ['onnx', 'torchscript']
    
    created_files = []
    
    # Export to different formats
    for format_name in formats:
        print(f"\n🔄 Exporting to {format_name.upper()}...")
        
        try:
            if format_name == 'onnx':
                exported_path = model.export(
                    format='onnx',
                    simplify=True,
                    imgsz=imgsz,
                    optimize=True
                )
                # Move to desired location with custom name
                target_path = output_dir / f"{base_name}.onnx"
                if exported_path != str(target_path):
                    if os.path.exists(exported_path):
                        os.rename(exported_path, target_path)
                created_files.append(str(target_path))
                
            elif format_name == 'torchscript':
                exported_path = model.export(
                    format='torchscript',
                    imgsz=imgsz,
                    optimize=True
                )
                target_path = output_dir / f"{base_name}.torchscript"
                if exported_path != str(target_path):
                    if os.path.exists(exported_path):
                        os.rename(exported_path, target_path)
                created_files.append(str(target_path))
                
            elif format_name == 'engine':  # TensorRT
                exported_path = model.export(
                    format='engine',
                    imgsz=imgsz,
                    half=True  # FP16 for better performance
                )
                target_path = output_dir / f"{base_name}.engine"
                if exported_path != str(target_path):
                    if os.path.exists(exported_path):
                        os.rename(exported_path, target_path)
                created_files.append(str(target_path))
                
            elif format_name == 'openvino':
                exported_path = model.export(
                    format='openvino',
                    imgsz=imgsz,
                    half=True
                )
                # OpenVINO creates a directory
                created_files.append(exported_path)
                
            print(f"   ✅ {format_name.upper()} export successful")
            
        except Exception as e:
            print(f"   ❌ {format_name.upper()} export failed: {e}")
    
    # Summary of created files
    print(f"\n📋 Created {len(created_files)} optimized models:")
    for filepath in created_files:
        if os.path.exists(filepath):
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath) / (1024*1024)
                print(f"   ✅ {filepath} ({size:.1f} MB)")
            else:
                print(f"   ✅ {filepath} (directory)")
        else:
            print(f"   ❌ {filepath} (not found)")
    
    return len(created_files) > 0

def find_test_image():
    """Find a suitable test image for benchmarking"""
    # Common test image locations
    test_paths = [
        "image.png", "test.png", "sample.png",
        "examples/0.png", "examples/ex0.png", "examples/image.png",
        "test.jpg", "sample.jpg", "image.jpg",
        "test_image.png", "benchmark.png"
    ]
    
    for path in test_paths:
        if os.path.exists(path):
            return path
    
    return None

def create_test_image(imgsz=320):
    """Create a realistic test image with some patterns"""
    # Create more realistic test image than pure random
    img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    
    # Add background
    img.fill(240)  # Light gray background
    
    # Add some geometric shapes to simulate manga content
    import cv2
    
    # Add rectangles (simulate speech bubbles)
    cv2.rectangle(img, (50, 50), (200, 150), (255, 255, 255), -1)  # White rectangle
    cv2.rectangle(img, (50, 50), (200, 150), (0, 0, 0), 2)        # Black border
    
    cv2.rectangle(img, (220, 180), (300, 250), (255, 255, 255), -1)
    cv2.rectangle(img, (220, 180), (300, 250), (0, 0, 0), 2)
    
    # Add some ellipses
    cv2.ellipse(img, (100, 200), (60, 40), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (100, 200), (60, 40), 0, 0, 360, (0, 0, 0), 2)
    
    return Image.fromarray(img)

def benchmark_models(model_paths, imgsz=320, runs=5):
    """
    Benchmark multiple model formats using real or synthetic test images
    """
    print(f"\n🚀 Benchmarking {len(model_paths)} models...")
    print(f"   Image size: {imgsz}x{imgsz}")
    print(f"   Runs per model: {runs}")
    
    # Check for ONNX runtime
    onnx_models = [p for p in model_paths if p.endswith('.onnx')]
    if onnx_models:
        try:
            import onnxruntime
            print(f"   ✅ ONNX Runtime: {onnxruntime.__version__}")
        except ImportError:
            print("   ⚠️  ONNX Runtime not found - install with: pip install onnxruntime")
    
    # Find or create test image
    test_image_path = find_test_image()
    if test_image_path:
        print(f"   📷 Using test image: {test_image_path}")
        try:
            test_image = Image.open(test_image_path).convert('RGB')
            # Resize if needed
            if test_image.size != (imgsz, imgsz):
                test_image = test_image.resize((imgsz, imgsz))
        except Exception as e:
            print(f"   ⚠️  Failed to load {test_image_path}: {e}")
            print("   🔧 Creating synthetic test image...")
            test_image = create_test_image(imgsz)
    else:
        print("   🔧 No test image found, creating synthetic test image...")
        test_image = create_test_image(imgsz)
    
    print("-" * 70)
    
    results = {}
    skipped = []
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"❌ {model_path}: File not found")
            skipped.append((model_path, "File not found"))
            continue
            
        model_name = Path(model_path).name
        print(f"\n🔍 Testing {model_name}...")
        
        # Show file info
        size_mb = os.path.getsize(model_path) / (1024*1024)
        print(f"   📊 Size: {size_mb:.1f} MB")
        
        try:
            # Special check for ONNX models
            if model_path.endswith('.onnx'):
                try:
                    import onnxruntime
                except ImportError:
                    print(f"   ❌ Skipping ONNX - onnxruntime not installed")
                    skipped.append((model_path, "ONNX Runtime missing"))
                    continue
            
            # Load model
            print("   📦 Loading model...")
            model = YOLO(model_path)
            print("   ✅ Model loaded successfully")
            
            # Warmup runs
            print("   🔥 Warming up (3 runs)...")
            warmup_times = []
            for i in range(3):
                try:
                    start = time.time()
                    _ = model(test_image, imgsz=imgsz, verbose=False)
                    warmup_time = time.time() - start
                    warmup_times.append(warmup_time)
                    print(f"      Warmup {i+1}: {warmup_time:.3f}s")
                except Exception as warmup_e:
                    print(f"      ❌ Warmup {i+1} failed: {warmup_e}")
                    
            if not warmup_times:
                raise Exception("All warmup runs failed")
            
            # Benchmark runs
            print("   ⏱️  Running benchmark...")
            times = []
            detections_counts = []
            
            for i in range(runs):
                try:
                    start_time = time.time()
                    result = model(test_image, imgsz=imgsz, verbose=False)
                    end_time = time.time()
                    
                    inference_time = end_time - start_time
                    times.append(inference_time)
                    
                    # Count detections
                    if hasattr(result[0], 'boxes') and result[0].boxes is not None:
                        detections = len(result[0].boxes)
                    else:
                        detections = 0
                    detections_counts.append(detections)
                    
                    print(f"      Run {i+1}/{runs}: {inference_time:.3f}s ({detections} detections)")
                    
                except Exception as run_e:
                    print(f"      ❌ Run {i+1} failed: {run_e}")
            
            if not times:
                raise Exception("All benchmark runs failed")
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            avg_detections = sum(detections_counts) / len(detections_counts) if detections_counts else 0
            
            results[model_name] = {
                'avg': avg_time,
                'min': min_time,
                'max': max_time,
                'path': model_path,
                'avg_detections': avg_detections,
                'warmup_avg': sum(warmup_times) / len(warmup_times) if warmup_times else 0
            }
            
            print(f"   📊 Results:")
            print(f"      Average: {avg_time:.3f}s ({1/avg_time:.1f} FPS)")
            print(f"      Range: {min_time:.3f}s - {max_time:.3f}s")
            print(f"      Avg detections: {avg_detections:.1f}")
            print(f"      Warmup avg: {results[model_name]['warmup_avg']:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            skipped.append((model_path, str(e)))
            
            # ONNX specific troubleshooting
            if model_path.endswith('.onnx'):
                print("   💡 ONNX troubleshooting:")
                print("      1. Install: pip install onnxruntime")
                print("      2. Re-export ONNX model")
                print("      3. Check model compatibility")
    
    # Show skipped models
    if skipped:
        print(f"\n⚠️  Skipped {len(skipped)} models:")
        for model_path, reason in skipped:
            model_name = Path(model_path).name
            print(f"   ❌ {model_name}: {reason}")
    
    # Results summary
    if results:
        print("\n" + "="*80)
        print("📈 BENCHMARK SUMMARY")
        print("="*80)
        
        # Sort by average time (fastest first)
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg'])
        fastest_time = sorted_results[0][1]['avg']
        
        print(f"{'Model':<25} {'Time (s)':<10} {'FPS':<8} {'Speedup':<8} {'Detections':<12}")
        print("-" * 80)
        
        for model_name, data in sorted_results:
            speedup = fastest_time / data['avg'] if data['avg'] > 0 else 0
            fps = 1 / data['avg']
            status = "🏆" if data['avg'] == fastest_time else "  "
            
            print(f"{status} {model_name:<23} {data['avg']:<10.3f} {fps:<8.1f} {speedup:<8.2f}x {data['avg_detections']:<12.1f}")
        
        print("\n🎯 Recommendations:")
        best_model = sorted_results[0]
        print(f"   🏆 Fastest: {best_model[0]} ({best_model[1]['avg']:.3f}s)")
        print(f"   💡 Update your app: MODEL = \"{best_model[1]['path']}\"")
        
        # Check detection consistency
        detection_counts = [data['avg_detections'] for _, data in sorted_results]
        if detection_counts and max(detection_counts) - min(detection_counts) > 1:
            print(f"   ⚠️  Detection counts vary ({min(detection_counts):.1f}-{max(detection_counts):.1f})")
            print("   💡 This is normal for different optimization formats")
        
        return best_model[1]['path']
    
    print(f"\n❌ No models successfully benchmarked!")
    return None

def find_related_models(base_model_path):
    """
    Find all related model files (different formats of the same model)
    """
    base_path = Path(base_model_path)
    base_name = base_path.stem
    search_dir = base_path.parent
    
    # Common YOLO export extensions
    extensions = ['.pt', '.onnx', '.torchscript', '.engine']
    
    related_models = []
    for ext in extensions:
        model_path = search_dir / f"{base_name}{ext}"
        if model_path.exists():
            related_models.append(str(model_path))
    
    return related_models

def main():
    parser = argparse.ArgumentParser(
        description="Universal YOLO Model Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python onix.py yolov8n.pt                    # Basic optimization
  python onix.py yolov11s.pt --imgsz 416       # Custom image size  
  python onix.py custom.pt --formats onnx      # Only ONNX export
  python onix.py model.pt --benchmark          # Just benchmark
  python onix.py yolov8m.pt --output ./models  # Custom output directory
        """)
    
    parser.add_argument('model_path', help='Path to the YOLO model (.pt file)')
    parser.add_argument('--formats', type=str, default='onnx,torchscript',
                       help='Export formats (comma-separated): onnx,torchscript,engine,openvino')
    parser.add_argument('--imgsz', type=int, default=320,
                       help='Image size for optimization (default: 320)')
    parser.add_argument('--output', type=str, help='Output directory for optimized models')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark on all related models')
    parser.add_argument('--benchmark-only', action='store_true',
                       help='Only run benchmark, skip optimization')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of benchmark runs per model (default: 5)')
    
    args = parser.parse_args()
    
    # Validate model file
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Parse formats
    formats = [f.strip().lower() for f in args.formats.split(',')]
    valid_formats = ['onnx', 'torchscript', 'engine', 'openvino']
    formats = [f for f in formats if f in valid_formats]
    
    if not formats and not args.benchmark_only:
        print(f"❌ Error: No valid formats specified. Valid: {valid_formats}")
        sys.exit(1)
    
    print("🎯 UNIVERSAL YOLO MODEL OPTIMIZER")
    print("="*50)
    print(f"📦 Input model: {args.model_path}")
    print(f"📐 Image size: {args.imgsz}x{args.imgsz}")
    
    success = True
    
    # Step 1: Create optimized models (unless benchmark-only)
    if not args.benchmark_only:
        print(f"🔧 Export formats: {', '.join(formats)}")
        success = create_optimized_models(args.model_path, args.output, formats, args.imgsz)
    
    # Step 2: Benchmark if requested or if optimization was successful
    if args.benchmark or args.benchmark_only:
        related_models = find_related_models(args.model_path)
        if related_models:
            best_model = benchmark_models(related_models, args.imgsz, args.runs)
            if best_model:
                print(f"\n💡 To use the best model in your app, update MODEL to:")
                print(f"   MODEL = \"{best_model}\"")
        else:
            print("❌ No related models found for benchmarking")
    
    print("\n" + "="*50)
    if success or args.benchmark_only:
        print("✅ PROCESS COMPLETE!")
        if not args.benchmark_only:
            print("🚀 Your optimized models are ready!")
    else:
        print("❌ PROCESS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()