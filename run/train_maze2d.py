import gflower.utils as utils
from itertools import cycle
import torch, os, tyro
from gflower.config.flow_matching import FlowMatchingTrainingConfig
from gflower.datasets.sequence import GoalDataset
from gflower.models_flow.transformer import TransformerFlow
from gflower.models_flow.flow_matcher import FlowMatcher
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tqdm
from gflower.utils.arrays import batch_to_device
from run.utils import deterministic, set_cuda_visible_device, save_config
from gflower.utils.debug import collate_fn

def train(cfg: FlowMatchingTrainingConfig, log_subfolder: str):
    if cfg.is_high_precision:
        torch.set_default_dtype(torch.float64)
    
    # goal dataset that fix start position and goal position, and only learn the flow between them. This is the same as CFM paper, and also makes training much easier.
    dataset = GoalDataset(
            env=cfg.env,
            horizon=cfg.horizon,
            normalizer=cfg.normalizer,
            preprocess_fns=cfg.preprocess_fns,
            max_path_length=cfg.max_path_length,
            seed=cfg.seed,
        )
    
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    
    transformer = TransformerFlow(
        seq_len=cfg.horizon,
        in_channels=action_dim + observation_dim,
        out_channels=action_dim + observation_dim,
        hidden_size=cfg.transformer_config.hidden_size,
        depth=cfg.transformer_config.depth,
        num_heads=cfg.transformer_config.num_heads,
        mlp_ratio=cfg.transformer_config.mlp_ratio,
        x_emb_proj=cfg.transformer_config.x_emb_proj,
        x_emb_proj_conv_k=cfg.transformer_config.x_emb_proj_conv_k,
    ).to(cfg.device)
    
    if cfg.is_high_precision:
        transformer = transformer.double()
    
    flow_matcher = FlowMatcher(
        action_dim=action_dim,
        model=transformer,
        flow_matching_type=cfg.flow_matching_type,
    )

    train_loader = cycle(DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True
    ))

    optimizer = torch.optim.Adam(transformer.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.save_freq * 2)
    ema = torch.optim.swa_utils.AveragedModel(
        transformer, 
        avg_fn=lambda avg, new, num: cfg.ema_decay * avg + (1 - cfg.ema_decay) * new
    )
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(log_subfolder, 'tensorboard_logs'))
    
    for i in tqdm.tqdm(range(cfg.n_train_steps)):
        batch = batch_to_device(next(train_loader), cfg.device) # (B, T, C)
        
        if cfg.is_high_precision:
            # Ensure batch is in double precision
            if isinstance(batch, (tuple, list)):
                batch = tuple(b.double() if torch.is_tensor(b) else b for b in batch)
            else:
                batch = batch.double()
        
        loss, infos = flow_matcher.loss(*batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Update EMA
        ema.update_parameters(transformer)

        if i % cfg.save_freq == 0:
            # Save both regular model and EMA model
            torch.save(
                transformer.state_dict(), 
                os.path.join(log_subfolder, f'model_{i // cfg.save_freq}.pth')
            )
            torch.save(
                ema.module.state_dict(), 
                os.path.join(log_subfolder, f'model_ema_{i // cfg.save_freq}.pth')
            )
            print("iter:",i,"loss info=",infos)
        writer.add_scalar('loss_moreStep', loss, i)


if __name__ == "__main__":
    cfg = tyro.cli(FlowMatchingTrainingConfig)
    
    set_cuda_visible_device(cfg)
    deterministic(cfg.seed) # seed everything

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, 'flow', f"{cfg.exp_name}_Hor_{cfg.horizon}")
    save_config(cfg, log_subfolder)

    train(cfg, log_subfolder)
