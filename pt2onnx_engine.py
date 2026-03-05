"""
Export YOLO .pt → .onnx + .engine (TensorRT) with maximum performance optimizations.

Usage:
    python pt2onnx.py                          # ใช้ค่า default (FP16, imgsz=1280)
    python pt2onnx.py --model weights/yolo.pt --int8 --calib data/coco8.yaml
    python pt2onnx.py --static                 # ใช้ static shape (เร็วขึ้นแต่ไม่ยืดหยุ่น)

Output:
    - weights/pill-detection-best.onnx   (FP16/INT8, optimized)
    - weights/pill-detection-best.engine (TensorRT, FP16/INT8, max perf)
"""
import argparse
import time
import sys
from pathlib import Path
from ultralytics import YOLO
import torch


def export_onnx(
    model: YOLO,
    model_path: str,
    output_dir: Path,
    imgsz: int,
    half: bool,
    int8: bool,
    calib_data: str,
    simplify: bool,
    opset: int,
    dynamic: bool,
) -> Path:
    """Export to ONNX with optimizations"""
    print(f"\n📦 Exporting ONNX: {model_path} → .onnx")
    print(f"   imgsz={imgsz}, half={half}, int8={int8}, dynamic={dynamic}")
    
    start = time.time()
    
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half and not int8,  # ถ้าใช้ int8 ไม่ต้องใส่ half
        int8=int8,
        data=calib_data if int8 else None,
        simplify=simplify,
        opset=opset,
        dynamic=dynamic,
        project=str(output_dir),
        name="",  # ใช้ชื่อเดียวกับไฟล์ต้นฉบับ
        verbose=True,
        device=0 if torch.cuda.is_available() else "cpu",
    )
    
    elapsed = time.time() - start
    print(f"✅ ONNX Exported: {onnx_path} ({elapsed:.1f}s)")
    return Path(onnx_path)


def export_tensorrt(
    model: YOLO,
    model_path: str,
    output_dir: Path,
    imgsz: int,
    half: bool,
    int8: bool,
    calib_data: str,
    workspace: int,
    static: bool,
    opset: int,
) -> Path:
    """Export to TensorRT Engine with maximum performance"""
    print(f"\n🔥 Exporting TensorRT: {model_path} → .engine")
    print(f"   imgsz={imgsz}, half={half}, int8={int8}, static={static}, workspace={workspace}GiB")
    
    # ตรวจสอบ prerequisites
    if not torch.cuda.is_available():
        raise RuntimeError("❌ TensorRT export ต้องการ GPU + CUDA")
    
    try:
        import tensorrt as trt  # noqa: F401
    except ImportError:
        print("⚠️  tensorrt ไม่พบใน environment")
        print("💡 ติดตั้งด้วย: pip install tensorrt หรือใช้ Docker: nvcr.io/nvidia/tensorrt")
        # พยายาม export ต่อไป ultralytics อาจจัดการให้
        print("🔄 ลอง export ต่อไป (ultralytics จะพยายามติดตั้ง/auto-detect)...")
    
    start = time.time()
    
    engine_path = model.export(
        format="engine",
        imgsz=imgsz,
        half=half and not int8,
        int8=int8,
        data=calib_data if int8 else None,
        workspace=workspace,      # GPU workspace สำหรับ layer fusion (ยิ่งมากยิ่ง optimize ได้ดี)
        dynamic=not static,       # static shape เร็วกว่า ~10-20% แต่ไม่ยืดหยุ่น
        simplify=True,            # ลด node ที่ไม่จำเป็นก่อน build engine
        opset=opset,
        project=str(output_dir),
        name="",
        verbose=True,
        device=0,
        # TensorRT-specific optimizations
        batch=1,                  # batch=1 เหมาะกับ realtime inference
        # ถ้า static=True จะ lock input shape ทำให้ TensorRT optimize ได้เต็มที่
    )
    
    elapsed = time.time() - start
    print(f"✅ TensorRT Engine Exported: {engine_path} ({elapsed:.1f}s)")
    return Path(engine_path)


def export_both(
    model_path: str = "weights/pill-detection-best-1.pt",
    imgsz: int = 640,
    half: bool = True,
    int8: bool = False,
    calib_data: str = "coco8.yaml",
    simplify: bool = True,
    opset: int = 17,
    dynamic: bool = False,      # Default เป็น static เพื่อความเร็วสูงสุด
    workspace: int = 8,         # 8 GiB workspace สำหรับ TensorRT
    output_dir: str = None,
) -> tuple[Path, Path]:
    """Export both ONNX and TensorRT formats"""
    
    # ตรวจสอบไฟล์ต้นฉบับ
    pt_path = Path(model_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"❌ Model not found: {pt_path}")
    
    # ตั้งค่า output directory
    if output_dir is None:
        output_dir = pt_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 YOLO Export Pipeline — Maximum Performance Mode")
    print(f"{'='*60}")
    print(f"📁 Input:  {pt_path.resolve()}")
    print(f"📁 Output: {output_dir.resolve()}")
    print(f"📐 imgsz:  {imgsz} {'(static)' if not dynamic else '(dynamic)'}")
    print(f"🔢 Precision: {'INT8' if int8 else 'FP16' if half else 'FP32'}")
    if int8:
        print(f"📊 Calib data: {calib_data}")
    print(f"{'='*60}\n")
    
    # โหลดโมเดล
    print("🔄 Loading YOLO model...")
    model = YOLO(str(pt_path))
    
    # Export ONNX (สำหรับ flexibility / cross-platform)
    onnx_path = export_onnx(
        model=model,
        model_path=str(pt_path),
        output_dir=output_dir,
        imgsz=imgsz,
        half=half,
        int8=int8,
        calib_data=calib_data,
        simplify=simplify,
        opset=opset,
        dynamic=dynamic,
    )
    
    # Export TensorRT (สำหรับความเร็วสูงสุดบน NVIDIA GPU)
    engine_path = export_tensorrt(
        model=model,
        model_path=str(pt_path),
        output_dir=output_dir,
        imgsz=imgsz,
        half=half,
        int8=int8,
        calib_data=calib_data,
        workspace=workspace,
        static=not dynamic,
        opset=opset,
    )
    
    # สรุปผล
    print(f"\n{'='*60}")
    print(f"✨ Export Complete!")
    print(f"{'='*60}")
    print(f"📦 ONNX:     {onnx_path.resolve()}")
    print(f"🔥 Engine:   {engine_path.resolve()}")
    print(f"\n💡 Usage in YOLOTracking:")
    print(f"   • ใช้ .engine สำหรับ deploy บน GPU NVIDIA (เร็วสุด)")
    print(f"   • ใช้ .onnx สำหรับ cross-platform / CPU fallback")
    print(f"{'='*60}\n")
    
    return onnx_path, engine_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export YOLO .pt → .onnx + .engine (TensorRT) — Max Performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model paths
    parser.add_argument(
        "--model", 
        default="weights/detection/pill-detection-best-1.pt",
        help="Path to input .pt model"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output directory (default: same as input model)"
    )
    
    # Image settings
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=1280,
        help="Image size for export (long edge)"
    )
    
    # Precision settings
    parser.add_argument(
        "--half", 
        action="store_true",
        default=True,  # Default เป็น FP16
        help="Export with FP16 precision (faster, less memory)"
    )
    parser.add_argument(
        "--no-half", 
        action="store_true",
        help="Use FP32 instead of FP16 (more accurate, slower)"
    )
    parser.add_argument(
        "--int8", 
        action="store_true",
        help="Export with INT8 quantization (fastest, requires calibration)"
    )
    parser.add_argument(
        "--calib-data", 
        type=str, 
        default="coco8.yaml",
        help="Dataset YAML for INT8 calibration (only if --int8)"
    )
    
    # Export options
    parser.add_argument(
        "--opset", 
        type=int, 
        default=17,
        help="ONNX opset version"
    )
    parser.add_argument(
        "--no-simplify", 
        action="store_true",
        help="Skip onnx-simplifier (not recommended)"
    )
    parser.add_argument(
        "--dynamic", 
        action="store_true",
        help="Use dynamic input shape (flexible but ~10-20% slower than static)"
    )
    
    # TensorRT options
    parser.add_argument(
        "--workspace", 
        type=int, 
        default=8,
        help="TensorRT workspace size in GiB (larger = better optimization, more VRAM)"
    )
    
    args = parser.parse_args()
    
    # Handle precision flags
    half = args.half and not args.no_half and not args.int8
    int8 = args.int8
    
    if int8 and not args.calib_data:
        print("❌ --int8 ต้องระบุ --calib-data สำหรับ calibration dataset")
        sys.exit(1)
    
    try:
        export_both(
            model_path=args.model,
            output_dir=args.output,
            imgsz=args.imgsz,
            half=half,
            int8=int8,
            calib_data=args.calib_data,
            simplify=not args.no_simplify,
            opset=args.opset,
            dynamic=args.dynamic,
            workspace=args.workspace,
        )
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)