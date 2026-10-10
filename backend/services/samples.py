import os

class SampleDataService:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.test_img_dir = os.path.join(self.base_dir, "dataset_clean", "floating_waste_v10", "test", "images")
        
    def get_sample_images(self, limit: int = 6):
        samples = []
        if os.path.exists(self.test_img_dir):
            files = sorted(os.listdir(self.test_img_dir))[:limit]
            for idx, fname in enumerate(files):
                fpath = os.path.join(self.test_img_dir, fname)
                sample_id = f"sample_{idx+1}"
                samples.append({
                    "id": sample_id,
                    "name": f"Marine Waste Sample {idx+1}",
                    "filename": fname,
                    "image_url": f"/api/samples/{sample_id}/image",
                    "filepath": fpath
                })
        else:
            # Fallback to test.jpg if available
            test_jpg = os.path.join(self.base_dir, "test.jpg")
            if os.path.exists(test_jpg):
                samples.append({
                    "id": "sample_1",
                    "name": "Default Test Image",
                    "filename": "test.jpg",
                    "image_url": "/api/samples/sample_1/image",
                    "filepath": test_jpg
                })
        return samples

    def get_sample_bytes(self, sample_id: str):
        samples = self.get_sample_images(limit=50)
        target = next((s for s in samples if s["id"] == sample_id or s["filename"] == sample_id), None)
        if not target:
            raise FileNotFoundError(f"Sample '{sample_id}' not found.")
            
        with open(target["filepath"], "rb") as f:
            return f.read(), target["filename"]

sample_service = SampleDataService()

