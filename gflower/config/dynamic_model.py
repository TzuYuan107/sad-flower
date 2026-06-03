from dataclasses import dataclass, field

@dataclass
class DynamicModelConfig:
    # general
    seed: int = 0
    device: str = 'cuda'
    log_folder: str = 'logs'
    exp_name: str = 'general'

    # environment
    env: str = 'maze2d-umaze-v1'
    horizon: int = 20 # transformer supports almost arbitrary horizon length
    normalizer: str = 'LimitsNormalizer'
    preprocess_fns: list = field(default_factory=lambda: [])
    max_path_length: int = 100000
    max_n_episodes: int = 100000
    termination_penalty: float = 0

    state_dim: int = 11 # observation dim
    action_dim: int = 3 # action dim

    # model
    esemble_num: int = 5
    hidden_dim: int = 256

    # training
    n_train_steps: int = 100000
    save_freq: int = 5000
    batch_size: int = 256
    learning_rate: float = 2e-4
    lr_schdule_T: int = 10000

    ema_decay: float = 0.995
    
    # validation
    val_frac: float = 0.05
    val_freq: int = 5000
    
    is_high_precision: bool = False

