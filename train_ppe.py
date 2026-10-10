"""
Optional stronger-training path.

Your dataset should contain these classes:
0 person
1 helmet
2 high visibility safety vest
3 safety boot
4 glove

Ultralytics accepts YOLO-format datasets with a data.yaml file.
Example:
    python train_ppe.py --data datasets/ppe/data.yaml
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO("yolo26n.pt")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        name="netra_shield_ppe",
        patience=15,
        plots=True,
    )


if __name__ == "__main__":
    main()
