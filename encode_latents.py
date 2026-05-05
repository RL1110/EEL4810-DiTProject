import os
import torch
from torchvision import transforms
from diffusers import AutoencoderKL
from PIL import Image
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader
from utils import load_config

class ImageEncodeDataset(Dataset):
    """Dataset to load raw images and return them ready for batch VAE encoding."""
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
        
    def __getitem__(self, idx):
        img_path, class_name, out_path = self.img_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor, class_name, out_path, True
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            return torch.zeros((3, 256, 256)), class_name, out_path, False


def encode_dataset(config_path, splits=["train", "val", "test"], batch_size=32, num_workers=4):
    config = load_config(config_path)
    archive_dir = os.path.dirname(config["dataset"]["train_dir"])
    cache_dir = config["dataset"]["cache_dir"]
    img_size = config["dataset"]["image_size"] # 256 default
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading VAE model: {config['vae']['model_id']}")
    vae = AutoencoderKL.from_pretrained(config["vae"]["model_id"]).to(device)
    vae.eval()
    
    # Image transformations: Resize, center crop, to tensor, normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    scaling_factor = vae.config.scaling_factor
    
    for split in splits:
        split_dir = os.path.join(archive_dir, split)
        out_split_dir = os.path.join(cache_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"Split directory {split_dir} does not exist. Skipping.")
            continue
            
        print(f"Encoding split {split}...")
        
        # Gathering all pending files
        img_paths = []
        for class_name in os.listdir(split_dir):
            class_path = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            out_class_path = os.path.join(out_split_dir, class_name)
            os.makedirs(out_class_path, exist_ok=True)
            
            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    out_file_name = os.path.splitext(file_name)[0] + ".pt"
                    out_path = os.path.join(out_class_path, out_file_name)
                    
                    if not os.path.exists(out_path):
                        img_paths.append((os.path.join(class_path, file_name), class_name, out_path))
        
        if not img_paths:
            print(f"No new images to encode for split {split}.")
            continue
            
        dataset = ImageEncodeDataset(img_paths, transform)
        # Using a DataLoader handles data fetching on multiple CPU threads avoiding bottleneck
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        
        with torch.no_grad():
            for batch_imgs, batch_classes, batch_out_paths, batch_valid in tqdm(dataloader):
                valid_mask = batch_valid == True
                if not valid_mask.any():
                    continue
                    
                valid_imgs = batch_imgs[valid_mask].to(device)
                
                # Batched VAE Encode
                latent_dist = vae.encode(valid_imgs).latent_dist
                # Deterministic mode
                latents = latent_dist.mode() * scaling_factor
                latents = latents.cpu()
                
                # Unzip and Save
                valid_indices = torch.where(valid_mask)[0].tolist()
                for i, idx in enumerate(valid_indices):
                    out_path = batch_out_paths[idx]
                    torch.save(latents[i].clone(), out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--split", type=str, help="Specific split to encode (e.g. train, val, test)", default=None)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for VAE encoding")
    parser.add_argument("--num_workers", type=int, default=6, help="Dataloader workers")
    
    args = parser.parse_args()
    splits = [args.split] if args.split else ["train", "val", "test"]
    
    encode_dataset(args.config, splits, args.batch_size, args.num_workers)
