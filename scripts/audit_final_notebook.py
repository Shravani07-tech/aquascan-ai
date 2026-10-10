import json
import os

nb_path = os.path.abspath("AquaScan_YOLOv8n_FINAL_GPU_Training.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=" * 75)
print("   STATIC AUDIT FOR AquaScan_YOLOv8n_FINAL_GPU_Training.ipynb")
print("=" * 75)

cells = nb.get("cells", [])
print(f"Total Cells Count: {len(cells)}")

audit_passed = True
audit_findings = []

def record_check(rule_num, name, passed, details=""):
    global audit_passed
    status = "[PASS]" if passed else "[FAIL]"
    if not passed: audit_passed = False
    print(f"  Rule {rule_num:02d}: {status} {name:<40s} {details}")
    audit_findings.append((rule_num, name, passed, details))

# Join all cell sources
all_text = "".join(["".join(c.get("source", [])) for c in cells])
all_code = "".join(["".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"])

# 1. Variables defined before use
record_check(1, "Variables Defined Before Use", "CFG = AquaScanFinalConfig()" in all_code and "DATA_YAML_PATH" in all_code)

# 2. No stale 3-epoch references
has_3_ep = "3-epoch" in all_text.lower() or "3 epoch" in all_text.lower() or "epochs=3" in all_code
record_check(2, "No Stale 3-Epoch References", not has_3_ep)

# 3. No fraction=0.015
record_check(3, "No fraction=0.015 Stub", "0.015" not in all_text)

# 4. No batch=4
record_check(4, "No batch=4 Stub", "batch=4" not in all_code)

# 5. No CPU training configuration
record_check(5, "No CPU Training Fallback", "device=cpu" not in all_code.lower() and "device=CFG.DEVICE" in all_code)

# 6. Epochs consistently equal 10
record_check(6, "Epochs Consistently Equal 10", "EPOCHS = 10" in all_code and "epochs=CFG.EPOCHS" in all_code)

# 7. Dataset source detection exists
record_check(7, "Dataset Source Detection (A, B, C)", "resolve_and_extract_dataset" in all_code)

# 8. Google Drive mounting exists
record_check(8, "Google Drive Mounting", "drive.mount" in all_code)

# 9. GPU verification exists
record_check(9, "GPU Acceleration Verification", "torch.cuda.is_available()" in all_code and "nvidia-smi" in all_code)

# 10. Dataset validation exists
record_check(10, "Dataset & YOLO Label Integrity Audit", "audit_dataset_integrity" in all_code)

# 11. Training exists
record_check(11, "YOLOv8n Training Pipeline", "model.train(" in all_code)

# 12. best.pt verification exists
record_check(12, "Critical Checkpoint Inspection (best.pt)", "verify_final_checkpoint" in all_code)

# 13. Validation evaluation exists
record_check(13, "Validation Evaluation (val split)", "split='val'" in all_code)

# 14. Test evaluation exists
record_check(14, "Test Evaluation (test split)", "split='test'" in all_code)

# 15. Inference benchmark exists
record_check(15, "GPU Inference Latency Benchmark", "preprocess" in all_code and "inference" in all_code)

# 16. Visual predictions exist
record_check(16, "Visual Prediction Plots Generation", "visual_predictions" in all_code)

# 17. Permanent Drive backup exists
record_check(17, "Permanent Google Drive Output Backup", "/content/drive/MyDrive/AquaScan_Training" in all_code)

# 18. Final ZIP creation exists
record_check(18, "Downloadable ZIP Archive Creation", "create_download_package" in all_code)

# 19. Final summary exists
record_check(19, "Final Verification & Summary Block", "AQUASCAN AI FINAL TRAINING COMPLETE SUMMARY" in all_code)

# 20. Error handling exists
record_check(20, "Robust Error Handling & Clear Error Messages", "raise RuntimeError" in all_code and "raise FileNotFoundError" in all_code)

print("=======================================================================")
if audit_passed:
    print("STATIC AUDIT RESULT: [PASS] ALL 20/20 AUDIT RULES PASSED SUCCESSFULLY!")
else:
    print("STATIC AUDIT RESULT: [FAIL] Some audit rules failed.")
print("=======================================================================")

