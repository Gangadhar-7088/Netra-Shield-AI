"""
Optional stronger-training path.

Your dataset should contain:
0 smoke
1 fire

Then run:
    python train_hazard.py --data datasets/hazard/data.yaml
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO("yolo26n.pt")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        name="netra_shield_hazard",
        patience=15,
        plots=True,
    )


if __name__ == "__main__":
    main()
