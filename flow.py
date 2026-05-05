import torch
from tqdm import tqdm

class RectifiedFlow:
    """
    Implements standard Rectified Flow formulation.
    x_1 = data
    x_0 = noise ~ N(0, I)
    x_t = t * x_1 + (1 - t) * x_0
    v_t = x_1 - x_0
    """
    def __init__(self):
        pass

    def compute_loss(self, model, x_1, y):
        """
        Computes the flow matching loss for a batch of data.
        x_1: Real latents [B, C, H, W]
        y: Class labels [B]
        """
        b, c, h, w = x_1.shape
        device = x_1.device
        
        # Sample x_0 (noise)
        x_0 = torch.randn_like(x_1)
        
        # Sample t ~ U(0, 1)
        t = torch.rand((b,), device=device)
        
        # Expand t to match image dims for interpolation
        t_expanded = t.view(b, 1, 1, 1)
        
        # Interpolate
        x_t = t_expanded * x_1 + (1.0 - t_expanded) * x_0
        
        # Target vector field
        target_v = x_1 - x_0
        
        # Predict vector field
        pred_v = model(x_t, t, y)
        
        # MSE Loss
        loss = torch.nn.functional.mse_loss(pred_v, target_v)
        return loss

    @torch.no_grad()
    def sample(self, model, b, y, device, num_steps=50, cfg_scale=1.0, latent_shape=(4, 32, 32)):
        """
        Sample from the model using Euler solver.
        """
        # Start from pure noise x_0
        shape = (b, *latent_shape)
        x = torch.randn(shape, device=device)
        
        # Time steps from 0 to 1
        d_t = 1.0 / num_steps
        
        for i in range(num_steps):
            # Current time t
            t_val = i / num_steps
            t = torch.full((b,), t_val, device=device)
            
            if cfg_scale > 1.0:
                # Need to run conditional and unconditional
                b_2 = b * 2
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t, t], dim=0)
                
                y_uncond = torch.full((b,), 100, device=device, dtype=torch.long)
                y_in = torch.cat([y, y_uncond], dim=0)
                
                v_pred = model(x_in, t_in, y_in)
                v_cond, v_uncond = v_pred.chunk(2, dim=0)
                
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(x, t, y)
                
            # Euler step
            x = x + v * d_t
            
        return x
