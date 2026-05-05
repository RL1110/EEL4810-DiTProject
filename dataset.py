import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class LatentDataset(Dataset):
    def __init__(self, data_dir, class_mapping=None, preload=True):
        """
        Args:
            data_dir (str): Path to the latents split directory (e.g., 'latents/train')
            class_mapping (dict): Mapping from class ID to integer label
            preload (bool): If True, load all latents into RAM to drastically unbottleneck GPU training.
        """
        self.data_dir = data_dir
        self.preload = preload
        self.latents = []
        self.labels = []
        self.files = []
        
        # If no mapping provided, build one by sorting class names
        class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        class_names.sort()
        
        if class_mapping is None:
            self.class_mapping = {class_id: i for i, class_id in enumerate(class_names)}
        else:
            self.class_mapping = class_mapping
            
        print(f"Indexing Dataset from {data_dir}...")
        for class_id in class_names:
            class_dir = os.path.join(data_dir, class_id)
            for f in os.listdir(class_dir):
                if f.endswith('.pt'):
                    self.files.append({
                        "path": os.path.join(class_dir, f),
                        "label": self.class_mapping[class_id]
                    })
                    
        if self.preload:
            print(f"Preloading {len(self.files)} latents into RAM (requires ~2GB). This may take a minute but completely eliminates IO bottlenecks...")
            # Using pre-allocated tensor can be more optimal, but a list of CPU tensors is fine for ~2GB.
            for item in tqdm(self.files, desc="Preloading Latents"):
                try:
                    latent = torch.load(item["path"], weights_only=True)
                    if isinstance(latent, torch.Tensor):
                        latent = latent.detach().clone()
                    self.latents.append(latent)
                    self.labels.append(item["label"])
                except Exception:
                    self.latents.append(torch.zeros((4, 32, 32)))
                    self.labels.append(item["label"])
                    
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        if self.preload:
            return self.latents[idx], self.labels[idx]
            
        item = self.files[idx]
        try:
            latent = torch.load(item["path"], weights_only=True)
            if isinstance(latent, torch.Tensor):
                latent = latent.detach().clone()
        except EOFError:
            latent = torch.zeros((4, 32, 32))
            
        return latent, item["label"]

def get_dataloader(data_dir, batch_size, class_mapping=None, shuffle=True, num_workers=4):
    # We load everything into RAM.
    dataset = LatentDataset(data_dir, class_mapping, preload=True)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=0, # RAM access is instant, workers just add communication overhead here
        pin_memory=True, # Keeps RAM contiguous for rapid GPU transfers
        drop_last=True if shuffle else False
    ), dataset.class_mapping
