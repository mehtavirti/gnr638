import os
import cv2
import time
import random

class ImageFolderDataset:
    def __init__(self, root_dir, image_size=32, preload=False):
        self.root = root_dir
        self.size = image_size
        self.class_map = {}
        self.image_paths = []
        self.labels = []
        self.preloaded_tensors = []
        
        self._index_data()
        
        if preload:
            self._preload_tensors()
    
    def _index_data(self):
        """Only scan folder structure and record file paths."""
        start = time.time()
        
        classes = sorted(os.listdir(self.root))
        for idx, cls in enumerate(classes):
            cls_path = os.path.join(self.root, cls)
            if not os.path.isdir(cls_path):
                continue
            self.class_map[cls] = idx
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_path, fname))
                    self.labels.append(idx)
        
        elapsed = time.time() - start
        print(f"Dataset loading time: {elapsed:.4f} seconds")
        print(f"Indexed {len(self.image_paths)} images across {len(self.class_map)} classes")
    
    def _preload_tensors(self):
        """Preload all images as C++ tensors - expensive but amortizes cost."""
        from framework.cpp_backend import Tensor
        
        print("Preloading images as C++ tensors (this will take ~10-15 minutes)...")
        start = time.time()
        
        for i, path in enumerate(self.image_paths):
            img_data = self._load_image(path)
            self.preloaded_tensors.append(Tensor(img_data))
            
            # Show progress every 1000 images for quick feedback
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (len(self.image_paths) - i - 1) / rate
                print(f"  Loaded {i+1}/{len(self.image_paths)} tensors | Speed: {rate:.0f} img/s | ETA: {remaining/60:.1f} min")
                
                # Early warning if too slow
                if i == 999 and rate < 50:
                    print(f"  ⚠️  WARNING: Speed is {rate:.0f} img/s (too slow!)")
                    print(f"  ⚠️  This will take {(len(self.image_paths) / rate) / 60:.0f} minutes to preload")
                    print(f"  ⚠️  Consider pressing Ctrl+C now and using subset training instead")
        
        elapsed = time.time() - start
        print(f"Preload complete! Took {elapsed/60:.1f} minutes ({len(self.image_paths)/elapsed:.0f} img/s average)")
    
    def _load_image(self, path):
        """Load and preprocess a single image."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.size, self.size))
        return [[float(img[r][c]) / 255.0 for c in range(self.size)] for r in range(self.size)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def get_batch(self, batch_size, shuffle=True):
        """Yield batches, using preloaded tensors if available."""
        indices = list(range(len(self.image_paths)))
        if shuffle:
            random.shuffle(indices)
        
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            
            if self.preloaded_tensors:
                # Use preloaded C++ tensors (fast)
                imgs = [self.preloaded_tensors[i] for i in batch_idx]
            else:
                # Load on-demand (slow)
                imgs = [self._load_image(self.image_paths[i]) for i in batch_idx]
            
            labels = [self.labels[i] for i in batch_idx]
            yield imgs, labels