from dataclasses import dataclass, field
from gflower.config.value import TransformerConfig as ValueTransformerConfig
from typing import List, Tuple

@dataclass
class TransformerConfig:
    hidden_size: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    x_emb_proj: str = 'conv'
    x_emb_proj_conv_k: int = 1

@dataclass
class FlowMatchingTrainingConfig:
    # general
    seed: int = 0
    device: str = 'cuda'
    log_folder: str = 'logs'
    exp_name: str = 'flow_matching'

    # environment
    env: str = 'maze2d-large-v1'
    horizon: int = 384 # transformer supports almost arbitrary horizon length
    normalizer: str = 'LimitsNormalizer'
    preprocess_fns: list = field(default_factory=lambda: [])
    max_path_length: int = 100000
    max_n_episodes: int = 100000
    termination_penalty: float = 0

    state_dim: int = 4 # observation dim
    action_dim: int = 2 # action dim

    # model
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    flow_matching_type: str = 'cfm' # 'cfm', 'ot_cfm', 'vp_cfm', 'sb_cfm'

    # training
    n_train_steps: int = 100000
    save_freq: int = 5000
    batch_size: int = 32
    learning_rate: float = 2e-4
    lr_schdule_T: int = 10000

    ema_decay: float = 0.995
    
    # base on pretrain result
    based_model_idx: int = 20
    base_exp_name: str = 'Friction_True'
    
    is_high_precision: bool = False

@dataclass
class FlowMatchingEvaluationConfig:
    # general
    seed: int = 0
    random_repeat: int = 5
    device: str = 'cuda'
    log_folder: str = 'logs'
    exp_name: str = 'flow_matching'
    
    #render
    IsRender: bool = True
    render_freq: int = 10
    
    #constraint
    obs_center: List[Tuple[float, float]] = field(default_factory=lambda: [(2, 1.5), (2, 2)])
    obs_radius: List[Tuple[float, float]] = field(default_factory=lambda: [(0.6, 1.2), (1.52, 1.52)])

    # environment
    env: str = 'maze2d-large-v1'
    horizon: int = 384
    normalizer: str = 'LimitsNormalizer'
    preprocess_fns: list = field(default_factory=lambda: [])
    max_path_length: int = 40000
    max_n_episodes: int = 100000
    max_episode_length: int = 1000 # max number of steps in one episode during evaluation
    termination_penalty: float = -100

    state_dim: int = 4 # observation dim
    action_dim: int = 2 # action dim

    # flow model
    transformer_config: TransformerConfig = field(default_factory=TransformerConfig)
    flow_exp_name: str = 'flow_matching'
    flow_cp: str = '0'
    flow_matching_type: str = 'cfm' # Must specify: this is necessary for MC guidance! 'cfm', 'ot_cfm', 'vp_cfm', 'sb_cfm'

    # value model
    value_exp_name: str = 'value'
    value_cp: str = '0'
    value_transformer_config: ValueTransformerConfig = field(default_factory=ValueTransformerConfig)

    # sampling
    batch_size: int = 1 # number of vetorized env
    ode_solver: str = 'euler'
    ode_t_span: Tuple[float, float] = (0.0, 1.0)
    ode_t_steps: int = 100

    guidance_method: str = 'gradient' # no, ss, gradient, mc, guidance_matching

    # sample and select
    ss_batch: int = 32 # how many candidate plans to sample for one env

    # value gudiance: gradient
    grad_scale: float = 1.0
    grad_schedule: str = 'const'
    grad_compute_at: str = 'x1' # 'x1', 'xt'
    grad_wrt: str = 'x1' # 'x1', 'xt'
    grad_precondition: str = 'none' # 'none', 'cov'

    # MC approximate guidance
    mc_batch_size: int = 64 # how support samples to calculate gradient for one env
    mc_scale: float = 1.0 # scale of the guidance
    mc_ep: float = 1e-2 # epsilon for numerical stability
    mc_ot_std: float = 0.2 # std of the OT-CFM
    mc_self_normalize: bool = False # whether to self normalize the value function
    mc_linear_J: bool = False # whether to use linear J for MC guidance
    mc_ss: int = 1
    
    # learned guidance
    guide_scale: float = 1.0 # scale of the guidance
    guide_model_transformer_config: TransformerConfig = field(
        default=TransformerConfig()
    )
    guide_matching_type: str = 'direct' # 'direct', 'use_learned_v', 'rw_use_learned_z', 'rw'
    guide_model_exp_name: str = ''
    guide_model_cp: str = '2'
    guide_inference_scale: float = 1.0 # scale of the inference

    # sim-mc guidance
    sim_mc_n: int = 100
    sim_mc_J_scale: float = 1.0 # scale of exp(-scale * J)
    sim_mc_std: float = 0.1

    sim_mc_schedule: str = 'linear_decay' # 'linear', 'cosine'
    sim_mc_scale: float = 1.0 # scale for guidance

    sim_mc_eps: float = 1e-2
    sim_mc_self_normalize: bool = True
    
    constraint_strategy: str = "No"
    act_pos_lim: float = 1.0
    act_neg_lim: float = -1.0
    
    # PTCLF-CBF setting
    start_time: float = 0.7
    CBF_c: Tuple[float, float] = (0.5, 0.5)
    CLF_c: float = 0.8
    PT_CBF_min: float = 0.005
    PT_CLF_min: float = 0.05
    PT_ord1: float = 10
    Stop_V: float = 0
    
    # NN_dynamics
    hidden_dim: int = 256
    NN_esemble_idx: int = 0
    NN_model_idx: int = 18
    NN_folder: str = "smooth"
    IsEma: bool = False
    
    #explicit dynamics (come from D4RL maze_model)
    point_mass: float = 4.18879 #radius=0.1,density=1000
    gear_ratio: float = 100
    viscious: float = 1
    sim_time: float = 0.01
    obs_max: List[float] = field(default_factory=lambda: [
        7.2163844, 10.219488, 
        5.2262554, 5.2262554
    ])
    obs_min: List[float] = field(default_factory=lambda: [
        0.39643136, 0.44179875, 
        -5.2262554, -5.2262554
    ])
    act_max: List[float] = field(default_factory=lambda: [
        1., 1.
    ])
    act_min: List[float] = field(default_factory=lambda: [
        -1., -1.
    ])
    
    # debug
    final_polciy_file_name: str = "final_policy_record"
    ACSCDCFlag: int = 1
    nEvaluaIdx: int = 0
    result_folder: str = "/workspace" 
    
    #robust term
    d_robust: float = 0
    s_robust: float = 0
    a_robust: float = 0
    
    # float64
    Is_Float64: bool = False
    