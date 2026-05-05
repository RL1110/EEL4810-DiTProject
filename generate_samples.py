import argparse
import os
import torch
from torchvision.utils import save_image
from diffusers import AutoencoderKL

from utils import load_config
from model import DiT
from flow import RectifiedFlow

@torch.no_grad()
def generate_samples(config_path, checkpoint_path, out_dir, num_samples=16, class_idx=None, cfg_scale=4.0):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(out_dir, exist_ok=True)
    num_classes = config["dataset"]["num_classes"]
    
    print("Loading model...")
    model = DiT(
        input_size=config["vae"]["latent_size"],
        patch_size=config["model"]["patch_size"],
        in_channels=config["vae"]["latent_channels"],
        hidden_size=config["model"]["hidden_dim"],
        depth=config["model"]["depth"],
        num_heads=config["model"]["num_heads"],
        num_classes=num_classes + 1
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    
    # Remove _orig_mod. prefix added by torch.compile during training
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model.eval()
    
    print(f"Loading VAE: {config['vae']['model_id']}")
    vae = AutoencoderKL.from_pretrained(config["vae"]["model_id"]).to(device)
    vae.eval()
    
    flow = RectifiedFlow()
    
    if class_idx is None:
        # Sample random classes
        classes = torch.randint(0, num_classes, (num_samples,), device=device)
    else:
        classes = torch.full((num_samples,), class_idx, device=device, dtype=torch.long)
        
    print(f"Sampling {num_samples} images with CFG scale {cfg_scale}...")
    
    latents = flow.sample(
        model, 
        b=num_samples, 
        y=classes, 
        device=device,
        num_steps=config["sampling"]["num_steps"],
        cfg_scale=cfg_scale,
        latent_shape=(config["vae"]["latent_channels"], config["vae"]["latent_size"], config["vae"]["latent_size"])
    )
    
    print("Decoding latents...")
    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1) # Unnormalize from [-1, 1] to [0, 1]
    
    out_path = os.path.join(out_dir, "samples.png")
    save_image(images, out_path, nrow=int(num_samples**0.5))
    print(f"Saved generated samples to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--out_dir", type=str, default="outputs/samples")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--class_idx", type=int, default=None, help="Specific class ID to generate. If None, random classes generated.")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    
    args = parser.parse_args()
    generate_samples(args.config, args.checkpoint, args.out_dir, args.num_samples, args.class_idx, args.cfg_scale)
