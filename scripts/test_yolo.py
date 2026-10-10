import os
import time
from ultralytics import YOLO

def main():
    image_path = 'test.jpg'
    output_dir = 'outputs/yolo_test'
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} does not exist.")
        return
    
    print("Loading YOLO11n model...")
    # This will load yolo11n.pt. If not present locally, Ultralytics downloads it.
    model = YOLO('yolo11n.pt')
    
    print(f"Running inference on {image_path}...")
    start_time = time.time()
    results = model(image_path)
    inference_time = time.time() - start_time
    
    print(f"Inference complete in {inference_time:.3f} seconds.\n")
    
    result = results[0]
    
    # Print detected objects, confidences, bounding boxes
    boxes = result.boxes
    if len(boxes) == 0:
        print("No objects detected.")
    else:
        print(f"{'Class':<15} | {'Confidence':<10} | {'Bounding Box (xyxy)':<30}")
        print("-" * 65)
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            # format coords
            xyxy_str = f"[{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]"
            print(f"{cls_name:<15} | {conf:<10.4f} | {xyxy_str:<30}")
    
    # Save annotated image
    output_path = os.path.join(output_dir, 'test_annotated.jpg')
    result.save(filename=output_path)
    print(f"\nAnnotated result saved to: {output_path}")

if __name__ == '__main__':
    main()
