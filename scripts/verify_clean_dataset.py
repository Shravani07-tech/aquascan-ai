import os
import yaml

def verify_dataset():
    clean_dir = "dataset_clean/floating_waste_v10"
    data_yaml_path = os.path.join(clean_dir, "data.yaml")
    
    print("--- 1. VERIFYING DATA.YAML ---")
    if not os.path.exists(data_yaml_path):
        print(f"ERROR: {data_yaml_path} does not exist!")
        return False
        
    with open(data_yaml_path, "r") as f:
        yaml_content = yaml.safe_load(f)
        
    print("data.yaml content:")
    for k, v in yaml_content.items():
        print(f"  {k}: {v}")
        
    splits = ["train", "valid", "test"]
    num_classes = yaml_content.get("nc", 5)
    class_names = yaml_content.get("names", [])
    
    print("\n--- 2. VERIFYING 5-FIELD BOUNDING BOXES IN ALL LABELS ---")
    total_files = 0
    total_boxes = 0
    invalid_boxes = 0
    empty_files = 0
    
    label_issues = []
    
    for split in splits:
        img_dir = os.path.join(clean_dir, split, "images")
        lbl_dir = os.path.join(clean_dir, split, "labels")
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"ERROR: Missing directory for split '{split}'")
            return False
            
        images = os.listdir(img_dir)
        labels = os.listdir(lbl_dir)
        
        print(f"Split '{split}': {len(images)} images, {len(labels)} label files")
        
        for img_name in images:
            base_name = os.path.splitext(img_name)[0]
            lbl_name = base_name + ".txt"
            lbl_path = os.path.join(lbl_dir, lbl_name)
            
            if not os.path.exists(lbl_path):
                label_issues.append(f"Missing label file for image: {img_name}")
                invalid_boxes += 1
                continue
                
            total_files += 1
            
            with open(lbl_path, "r") as lf:
                lines = lf.read().strip().split("\n")
                
            if not lines or (len(lines) == 1 and lines[0] == ""):
                empty_files += 1
                continue
                
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if not parts:
                    continue
                total_boxes += 1
                
                if len(parts) != 5:
                    label_issues.append(f"File {lbl_name} Line {line_idx+1}: Has {len(parts)} fields instead of 5")
                    invalid_boxes += 1
                    continue
                    
                try:
                    cid = int(float(parts[0]))
                    coords = [float(p) for p in parts[1:]]
                except ValueError:
                    label_issues.append(f"File {lbl_name} Line {line_idx+1}: Non-numeric values")
                    invalid_boxes += 1
                    continue
                    
                if cid < 0 or cid >= num_classes:
                    label_issues.append(f"File {lbl_name} Line {line_idx+1}: Class ID {cid} out of range [0, {num_classes-1}]")
                    invalid_boxes += 1
                    
                x_c, y_c, w, h = coords
                if not (0.0 <= x_c <= 1.0 and 0.0 <= y_c <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    label_issues.append(f"File {lbl_name} Line {line_idx+1}: Coords out of range [0,1]")
                    invalid_boxes += 1
                    
                if w <= 0 or h <= 0:
                    label_issues.append(f"File {lbl_name} Line {line_idx+1}: Non-positive dimension (w={w}, h={h})")
                    invalid_boxes += 1

    print("\n--- VERIFICATION RESULT ---")
    print(f"Total Label Files Verified: {total_files}")
    print(f"Total Bounding Boxes Checked: {total_boxes}")
    print(f"Empty Label Files (Backgrounds): {empty_files}")
    print(f"Invalid Bounding Boxes Found: {invalid_boxes}")
    
    if invalid_boxes == 0:
        print("SUCCESS: 100% of labels are valid 5-field YOLO bounding boxes!")
        return True
    else:
        print(f"WARNING: Found {len(label_issues)} issues:")
        for issue in label_issues[:10]:
            print(f"  - {issue}")
        return False

if __name__ == "__main__":
    verify_dataset()
