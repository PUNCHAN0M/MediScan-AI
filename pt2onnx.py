"""
Export YOLO .pt → .onnx with dynamic image size + optimizations.

Usage:
    python pt2onnx.py                          # ใช้ค่า default
    python pt2onnx.py --model model/yolo12-seg.pt --imgsz 640 --half
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def export_onnx(
    model_path: str = "model/pill-detection-best.pt",
    imgsz: int = 640,
    half: bool = False,
    simplify: bool = True,
    opset: int = 17,
    dynamic: bool = True,
) -> Path:

    model = YOLO(model_path)

    out = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        simplify=simplify,
        opset=opset,
        dynamic=dynamic,
        project=r"D:\project\Medicine-AI\MediScan-AI\model",
        name="",
    )

    print(f"\n✅ Exported: {out}")
    return Path(out)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO .pt → .onnx")
    parser.add_argument("--model", default="model/pill-detection-best.pt",
                        help="Path to .pt model")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Default image size for export shape")
    parser.add_argument("--half", action="store_true",
                        help="Export FP16 (requires GPU)")
    parser.add_argument("--no-simplify", action="store_true",
                        help="Skip onnx-simplifier")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-dynamic", action="store_true",
                        help="Fixed input size (ไม่แนะนำ)")
    args = parser.parse_args()

    export_onnx(
        model_path=args.model,
        imgsz=args.imgsz,
        half=args.half,
        simplify=not args.no_simplify,
        opset=args.opset,
        dynamic=not args.no_dynamic,
    )
