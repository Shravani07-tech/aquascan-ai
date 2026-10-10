import os
import zipfile
import time

def package_gpu_zip():
    zip_filename = "AquaScan_GPU_Training_Pack.zip"
    print(f"Creating GPU training zip package: {zip_filename}...")
    
    clean_dir = "dataset_clean/floating_waste_v10"
    files_to_pack = [
        "yolov8n.pt",
        "AquaScan_YOLOv8n_GPU_Training.ipynb",
        "scripts/train_yolov8_full_gpu.py"
    ]
    
    start_time = time.time()
    count = 0
    
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Pack individual files
        for fpath in files_to_pack:
            if os.path.exists(fpath):
                zipf.write(fpath, arcname=fpath)
                count += 1
                print(f"Packed file: {fpath}")
                
        # Pack dataset_clean directory
        print(f"Packing directory: {clean_dir}...")
        for root, dirs, files in os.walk(clean_dir):
            for file in files:
                filepath = os.path.join(root, file)
                # Ensure relative path inside zip matches structure
                arcname = filepath.replace("\\", "/")
                zipf.write(filepath, arcname=arcname)
                count += 1
                if count % 1000 == 0:
                    print(f"Packed {count} items...")
                    
    elapsed = time.time() - start_time
    zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print(f"\nSuccessfully created {zip_filename}!")
    print(f"Total items packed: {count}")
    print(f"Package Size: {zip_size_mb:.2f} MB")
    print(f"Packaging completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    package_gpu_zip()
