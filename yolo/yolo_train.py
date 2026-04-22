from ultralytics import YOLO
from pathlib import Path
import yaml

if __name__ == "__main__":
    root = Path(__file__).parent
    ds_path = root / 'ds'
    
    classes = {0: "cube", 1: "neither", 2: "sphere"}
    
    config = {
        "path": str(ds_path.absolute()),
        "train": str((ds_path / "images" / "train").absolute()),
        "val": str((ds_path / "images" / "val").absolute()),
        "nc": len(classes),
        "names": classes
    }
    
    with open(ds_path / "dataset.yaml", "w") as f:
        yaml.dump(config, f, allow_unicode=True)
    
    size = "n" # n, s, m, l, x
    model = YOLO(f"yolo26{size}.pt")
    
    result = model.train(
        data = str(ds_path / "dataset.yaml"),
        imgsz = 640,
        batch = 64,
        workers = 0,
        
        epochs = 50,
        patience = 5,
        optimizer = "AdamW",
        lr0 = 0.01,
        warmup_epochs = 3,
        cos_lr = True,
        dropout = 0.2,
        
        hsv_h = 0.015,
        hsv_s = 0.7,
        hsv_v = 0.4,
        flipud = 0.0,
        fliplr = 0.5,
        mosaic = 1.0,
        degrees = 5.0,
        scale = 0.5,
        translate = 0.1,
        
        conf = 0.001,
        iou = 0.7,
        
        project = "figures",
        name = "yolo",
        save = True,
        save_period = 5,
        # device = "cpu",
        device = "0",
        
        verbose = True,
        plots = True,
        val = True,
        close_mosaic = 8,
        amp = True # FP16
    )
    
    print("Done")
    print(result.save_dir)