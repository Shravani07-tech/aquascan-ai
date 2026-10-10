import os
import cv2
import hashlib
import yaml
from collections import defaultdict

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    dataset_dir = "dataset_raw/floating_waste_v10"
    splits = ["train", "valid", "test"]
    
    # Load class names from data.yaml if available, otherwise use defaults
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
            class_names = data_yaml.get('names', ['bottle', 'can', 'carton', 'paper', 'plastic'])
    else:
        class_names = ['bottle', 'can', 'carton', 'paper', 'plastic']
    
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes: {class_names}")

    # Task 1: Class Distribution
    # Map split -> class_id -> count
    obj_counts = {split: defaultdict(int) for split in splits}
    img_counts_by_class = {split: defaultdict(set) for split in splits} # split -> class_id -> set of image names
    
    # Task 2: Label Validation metrics
    problems_label = {
        "invalid_class_id": 0,
        "incorrect_fields_count": 0,
        "non_numeric_fields": 0,
        "out_of_normalized_range": 0,
        "width_or_height_zero_or_negative": 0,
        "box_outside_boundaries": 0,
        "missing_label_file": 0,
        "empty_label_file": 0,
        "corrupt_label_file": 0
    }
    
    # Task 3: Image Validation metrics
    problems_image = {
        "corrupt_image": 0,
        "unexpected_format": 0,
        "duplicate_filenames": defaultdict(int),
        "exact_duplicates_by_hash": defaultdict(list)
    }
    
    image_dims = [] # list of (w, h)
    
    # Scan all splits
    for split in splits:
        split_dir = os.path.join(dataset_dir, split)
        img_dir = os.path.join(split_dir, "images")
        lbl_dir = os.path.join(split_dir, "labels")
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"Warning: Directory missing for split {split}")
            continue
            
        images = os.listdir(img_dir)
        labels = os.listdir(lbl_dir)
        
        # Track duplicate filenames in this split (should be unique within split)
        seen_filenames_in_split = set()
        
        for img_name in images:
            img_path = os.path.join(img_dir, img_name)
            
            # Check duplicate filenames
            if img_name in seen_filenames_in_split:
                problems_image["duplicate_filenames"][img_name] += 1
            seen_filenames_in_split.add(img_name)
            
            # Image Format Check
            ext = os.path.splitext(img_name)[1].lower()
            if ext != ".jpg" and ext != ".jpeg":
                problems_image["unexpected_format"] += 1
                
            # Image Corruption & Dimensions Check
            try:
                img = cv2.imread(img_path)
                if img is None:
                    problems_image["corrupt_image"] += 1
                    continue
                h, w, c = img.shape
                image_dims.append((w, h))
                
                # MD5 hash for exact duplicates
                hsh = get_md5(img_path)
                problems_image["exact_duplicates_by_hash"][hsh].append(img_path)
            except Exception as e:
                problems_image["corrupt_image"] += 1
                continue
            
            # Find corresponding label file
            base_name = os.path.splitext(img_name)[0]
            lbl_name = base_name + ".txt"
            lbl_path = os.path.join(lbl_dir, lbl_name)
            
            if not os.path.exists(lbl_path):
                problems_label["missing_label_file"] += 1
                continue
                
            # Parse label file
            try:
                with open(lbl_path, "r") as lf:
                    lines = lf.read().strip().split("\n")
                    
                if not lines or (len(lines) == 1 and lines[0] == ""):
                    problems_label["empty_label_file"] += 1
                    continue
                    
                has_valid_box = False
                for line_idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) != 5:
                        problems_label["incorrect_fields_count"] += 1
                        continue
                    
                    try:
                        class_id_f = float(parts[0])
                        coords = [float(p) for p in parts[1:]]
                    except ValueError:
                        problems_label["non_numeric_fields"] += 1
                        continue
                        
                    if not class_id_f.is_integer():
                        problems_label["invalid_class_id"] += 1
                        continue
                    class_id = int(class_id_f)
                    
                    if class_id < 0 or class_id >= num_classes:
                        problems_label["invalid_class_id"] += 1
                        continue
                    
                    # Track class counts
                    obj_counts[split][class_id] += 1
                    img_counts_by_class[split][class_id].add(img_name)
                    
                    x_c, y_c, bw, bh = coords
                    # Check range [0, 1]
                    if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                        problems_label["out_of_normalized_range"] += 1
                        
                    # Check dimension > 0
                    if bw <= 0 or bh <= 0:
                        problems_label["width_or_height_zero_or_negative"] += 1
                        
                    # Check boundaries
                    x_min = x_c - bw / 2
                    y_min = y_c - bh / 2
                    x_max = x_c + bw / 2
                    y_max = y_c + bh / 2
                    
                    # With floating point threshold
                    tol = 1e-6
                    if (x_min < -tol or y_min < -tol or x_max > 1.0 + tol or y_max > 1.0 + tol):
                        problems_label["box_outside_boundaries"] += 1
            except Exception as e:
                problems_label["corrupt_label_file"] += 1
                
    # Compile Task 1 Distribution tables
    print("\n--- CLASS DISTRIBUTION RESULT ---")
    splits_data = {}
    for class_id in range(num_classes):
        class_name = class_names[class_id]
        splits_data[class_name] = {}
        total_objs = 0
        total_imgs = 0
        for split in splits:
            objs = obj_counts[split][class_id]
            imgs = len(img_counts_by_class[split][class_id])
            splits_data[class_name][f"{split}_objs"] = objs
            splits_data[class_name][f"{split}_imgs"] = imgs
            total_objs += objs
            # Note: total images containing the class across splits is the sum of split images, since splits are disjoint
            total_imgs += imgs
        splits_data[class_name]["total_objs"] = total_objs
        splits_data[class_name]["total_imgs"] = total_imgs
        
        print(f"Class {class_id} ({class_name}):")
        print(f"  Train: {splits_data[class_name]['train_objs']} objs in {splits_data[class_name]['train_imgs']} imgs")
        print(f"  Valid: {splits_data[class_name]['valid_objs']} objs in {splits_data[class_name]['valid_imgs']} imgs")
        print(f"  Test : {splits_data[class_name]['test_objs']} objs in {splits_data[class_name]['test_imgs']} imgs")
        print(f"  Total: {total_objs} objs in {total_imgs} imgs")

    # Image Dimension Statistics
    widths = [d[0] for d in image_dims]
    heights = [d[1] for d in image_dims]
    unique_resolutions = set(image_dims)
    
    print("\n--- IMAGE DIMENSION STATISTICS ---")
    print(f"Total valid images checked: {len(image_dims)}")
    print(f"Unique resolutions found: {unique_resolutions}")
    if widths:
        print(f"Widths: min={min(widths)}, max={max(widths)}")
        print(f"Heights: min={min(heights)}, max={max(heights)}")

    # Duplicates Analysis
    exact_duplicates = {k: v for k, v in problems_image["exact_duplicates_by_hash"].items() if len(v) > 1}
    num_duplicate_groups = len(exact_duplicates)
    total_duplicate_files = sum(len(v) for v in exact_duplicates.values())
    
    print(f"\n--- DUPLICATE IMAGES ---")
    print(f"Found {num_duplicate_groups} groups of exact duplicate images, total files: {total_duplicate_files}")

    # Task 4: Visual Sample Selection
    # Let's find one image for each class and draw annotations on it, then save it
    # We prefer train or valid split
    outputs_dir = "outputs/dataset_audit"
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Colors for the 5 classes
    colors = [
        (0, 0, 255),    # Red for bottle
        (0, 255, 0),    # Green for can
        (255, 0, 0),    # Blue for carton
        (0, 255, 255),  # Yellow for paper
        (255, 0, 255)   # Magenta for plastic
    ]
    
    representative_samples = {}
    for class_id in range(num_classes):
        class_name = class_names[class_id]
        # Search for an image containing this class
        found = False
        for split in ["valid", "train"]: # Search valid first since it's smaller, then train
            if found:
                break
            img_dir = os.path.join(dataset_dir, split, "images")
            lbl_dir = os.path.join(dataset_dir, split, "labels")
            
            for img_name in os.listdir(img_dir):
                base_name = os.path.splitext(img_name)[0]
                lbl_path = os.path.join(lbl_dir, base_name + ".txt")
                if not os.path.exists(lbl_path):
                    continue
                    
                # Read labels to check if it contains class_id
                with open(lbl_path, "r") as lf:
                    lines = lf.read().strip().split("\n")
                
                has_class = False
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            cid = int(float(parts[0]))
                            if cid == class_id:
                                has_class = True
                                break
                        except ValueError:
                            continue
                
                if has_class:
                    # Draw boxes on this image
                    img_path = os.path.join(img_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w, c = img.shape
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                try:
                                    cid = int(float(parts[0]))
                                    x_c, y_c, bw, bh = [float(p) for p in parts[1:]]
                                    
                                    # Convert to pixel coords
                                    x1 = int((x_c - bw / 2) * w)
                                    y1 = int((y_c - bh / 2) * h)
                                    x2 = int((x_c + bw / 2) * w)
                                    y2 = int((y_c + bh / 2) * h)
                                    
                                    # Draw bounding box
                                    color = colors[cid] if cid < len(colors) else (255, 255, 255)
                                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                                    
                                    # Label text
                                    label_name = class_names[cid]
                                    # Draw background rectangle for text
                                    (tw, th), _ = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                    cv2.rectangle(img, (x1, y1 - 20), (x1 + tw, y1), color, -1)
                                    cv2.putText(img, label_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                                except Exception:
                                    pass
                        
                        out_filename = f"sample_{class_name}.jpg"
                        out_path = os.path.join(outputs_dir, out_filename)
                        cv2.imwrite(out_path, img)
                        representative_samples[class_name] = out_path
                        print(f"Saved visual sample for {class_name} to {out_path}")
                        found = True
                        break

    # Write report.md
    report_path = os.path.join(outputs_dir, "report.md")
    with open(report_path, "w") as rf:
        rf.write("# AquaScan AI - Dataset Quality Audit Report\n\n")
        rf.write(f"This report presents the findings of the quality audit conducted on the **Floating Waste v10** dataset.\n\n")
        
        # Task 1: Class Distribution Table
        rf.write("## Task 1: Class Distribution\n\n")
        rf.write("| Class Name | Train Objs (Imgs) | Valid Objs (Imgs) | Test Objs (Imgs) | Total Objs (Imgs) |\n")
        rf.write("|---|---|---|---|---|\n")
        for class_id in range(num_classes):
            c_name = class_names[class_id]
            d = splits_data[c_name]
            rf.write(f"| **{c_name}** | {d['train_objs']} ({d['train_imgs']}) | {d['valid_objs']} ({d['valid_imgs']}) | {d['test_objs']} ({d['test_imgs']}) | {d['total_objs']} ({d['total_imgs']}) |\n")
        rf.write("\n")
        
        # Task 2: Label Validation
        rf.write("## Task 2: Label Validation\n\n")
        rf.write("All YOLO label files were checked for common format and integrity issues. Here is a summary of the findings:\n\n")
        rf.write("| Validation Check | Problems Found |\n")
        rf.write("|---|---|\n")
        for check_name, count in problems_label.items():
            readable_name = check_name.replace("_", " ").capitalize()
            rf.write(f"| {readable_name} | {count} |\n")
        rf.write("\n")
        
        # Task 3: Image Validation
        rf.write("## Task 3: Image Validation\n\n")
        rf.write("Images were validated for reading errors, duplicate content, and dimensions:\n\n")
        rf.write(f"- **Unreadable/Corrupt Images:** {problems_image['corrupt_image']}\n")
        rf.write(f"- **Unexpected Image Formats:** {problems_image['unexpected_format']}\n")
        rf.write(f"- **Duplicate Filenames (within splits):** {len(problems_image['duplicate_filenames'])}\n")
        rf.write(f"- **Exact Content Duplicates (MD5 Hash groups):** {num_duplicate_groups} groups (affecting {total_duplicate_files} files)\n")
        rf.write(f"- **Image Resolutions Found:** {list(unique_resolutions)}\n")
        rf.write(f"- **Dimensions:** Min Width = {min(widths)}, Max Width = {max(widths)}, Min Height = {min(heights)}, Max Height = {max(heights)}\n\n")
        
        # Task 4: Visual Sample Inspection
        rf.write("## Task 4: Visual Sample Inspection\n\n")
        rf.write("Representative annotated images for each class are saved in `outputs/dataset_audit/`:\n\n")
        for class_name, path in representative_samples.items():
            # Use absolute path for embedding in artifacts
            abs_path = os.path.abspath(path)
            rf.write(f"### Class: {class_name}\n")
            rf.write(f"![Sample {class_name}]({abs_path.replace('\\', '/')})\n\n")
            
    print(f"\nAudit complete. Report written to {report_path}")

if __name__ == "__main__":
    main()
