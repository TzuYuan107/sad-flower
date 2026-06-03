import torch
from torch.utils.data import DataLoader, random_split
from gflower.config.dynamic_model import DynamicModelConfig
import os
import tyro
from gflower.config.dynamic_model import DynamicModelConfig
from run.utils import deterministic, set_cuda_visible_device, save_config
from gflower.utils.arrays import batch_to_device
from gflower.datasets.sequence import DynamicSequenceDataset
from gflower.models_dynamic.NN_dynamics_matcher import NNDynamicsJacRegMatcher
from gflower.models_dynamic.maze2d_dynamics import Maze2dNNDynamicsModel
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
from gflower.utils.debug import collate_fn
import tqdm

# ---------------------------------------------
# Ensemble helper
# ---------------------------------------------
def make_ensemble(num_models, state_dim, act_dim, hidden_dim, device="cuda", is_high_precision:bool = False):
    if is_high_precision:
        return [Maze2dNNDynamicsModel(state_dim, act_dim, hidden_dim).to(device).double() for _ in range(num_models)]
    else:
        return [Maze2dNNDynamicsModel(state_dim, act_dim, hidden_dim).to(device) for _ in range(num_models)]
        
        
def train_forward_dynamics(cfg:DynamicModelConfig,
                           log_subfolder):
    
    if cfg.is_high_precision:
        torch.set_default_dtype(torch.float64)
        
    full_dataset = DynamicSequenceDataset(
        env=cfg.env,
        horizon=cfg.horizon,
        normalizer=cfg.normalizer,
        preprocess_fns=cfg.preprocess_fns,
        max_path_length=cfg.max_path_length,
        max_n_episodes=cfg.max_n_episodes,
        seed=cfg.seed,
    )
    state_dim = full_dataset.observation_dim
    action_dim = full_dataset.action_dim
    normalizer = full_dataset.normalizer
    n_val    = int(len(full_dataset) * cfg.val_frac)
    n_train  = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(cfg.seed))
    
    train_loader = cycle(DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True))
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    
    
    ensemble = make_ensemble(cfg.esemble_num, 
                             state_dim, action_dim, 
                             hidden_dim=cfg.hidden_dim, is_high_precision=cfg.is_high_precision)
    
    for idx_esemble, forward_model in enumerate(ensemble):
        forward_model_matcher = NNDynamicsJacRegMatcher(forward_model)
        forward_model.train()
        
        optimizer = torch.optim.Adam(forward_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.save_freq * 2)
        ema = torch.optim.swa_utils.AveragedModel(
            forward_model, 
            avg_fn=lambda avg, new, num: cfg.ema_decay * avg + (1 - cfg.ema_decay) * new
        )
        if cfg.is_high_precision:
            ema.module = ema.module.double()
        flow_matcher_ema = NNDynamicsJacRegMatcher(model=ema.module)
        
        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir=os.path.join(log_subfolder, 'tensorboard_logs'))
        
        for step in tqdm.tqdm(range(cfg.n_train_steps)):
            batch = batch_to_device(next(train_loader), cfg.device) # (B, T, C)
            
            # Ensure batch is in correct precision
            if cfg.is_high_precision:
                if isinstance(batch, (tuple, list)):
                    batch = tuple(b.double() if torch.is_tensor(b) else b for b in batch)
                else:
                    batch = batch.double()
            
            loss, infos = forward_model_matcher.loss(*batch)
            loss.backward()
            optimizer.step()
            ema.update_parameters(forward_model) 
            loss_ema, infos_ema = flow_matcher_ema.loss(*batch)
            scheduler.step()
            optimizer.zero_grad()
            writer.add_scalar(f'train/loss_model_{idx_esemble}', loss, step)
            writer.add_scalar(f'train/loss_model_forward_{idx_esemble}', infos["Forward_loss"], step)
            writer.add_scalar(f'train/loss_model_jacobian_{idx_esemble}', infos["Jac_loss"], step)
            
            writer.add_scalar(f'train/loss_ema_model_{idx_esemble}', loss_ema, step)
            writer.add_scalar(f'train/loss_ema_jacobian_{idx_esemble}', infos_ema["Forward_loss"], step)
            writer.add_scalar(f'train/loss_ema_jacobian_{idx_esemble}', infos_ema["Jac_loss"], step)
            
            if step % cfg.val_freq == 0:
                forward_model.eval()
                val_losses = []
                forward_losses = []
                jacobian_losses = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = batch_to_device(val_batch, cfg.device)
                        # Ensure validation batch is in correct precision
                        if cfg.is_high_precision:
                            if isinstance(val_batch, (tuple, list)):
                                val_batch = tuple(b.double() if torch.is_tensor(b) else b for b in val_batch)
                            else:
                                val_batch = val_batch.double()
                        val_loss, val_infos = forward_model_matcher.loss(*val_batch, compute_jacobian=False)
                        val_losses.append(val_loss.item())
                        forward_losses.append(val_infos["Forward_loss"])
                        jacobian_losses.append(val_infos["Jac_loss"])
                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_val_forward_loss = sum(forward_losses) / len(forward_losses)
                avg_val_jacobian_losses = sum(jacobian_losses) / len(jacobian_losses)
                writer.add_scalar(f'val/loss_model_{idx_esemble}', avg_val_loss, step)
                writer.add_scalar(f'val/loss_model_forward_{idx_esemble}', avg_val_forward_loss, step)
                writer.add_scalar(f'val/loss_model_jacobian_{idx_esemble}', avg_val_jacobian_losses, step)
                forward_model.train()
                
                print("train_loss:",infos, "train_ema:",infos_ema)
                print("val_loss:",avg_val_loss,"valia_forward_loss",avg_val_forward_loss, "valia_reg_loss",avg_val_jacobian_losses)

            if step % cfg.save_freq == 0:
                # Save both regular model and EMA model
                torch.save(
                    forward_model.state_dict(), 
                    os.path.join(log_subfolder, f'model_esemble_{idx_esemble}_{step // cfg.save_freq}.pth')
                )
                torch.save(
                    ema.module.state_dict(), 
                    os.path.join(log_subfolder, f'model_ema_esemble_{idx_esemble}_{step // cfg.save_freq}.pth')
                )
            
            

if __name__ == '__main__':
    cfg = tyro.cli(DynamicModelConfig)
    
    set_cuda_visible_device(cfg)
    deterministic(cfg.seed) # seed everything
    
    log_subfolder = os.path.join(cfg.log_folder, cfg.env, 'model_dynamics', cfg.exp_name)
    save_config(cfg, log_subfolder)
    
    train_forward_dynamics(cfg, log_subfolder)