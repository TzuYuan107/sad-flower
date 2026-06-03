import os
import importlib
import random
import numpy as np
import torch
from tap import Tap

from gflower.config.flow_matching import FlowMatchingEvaluationConfig

from .serialization import mkdir
from .git_utils import (
    get_git_rev,
    save_git_diff,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def watch(args_to_watch):
    def _fn(args):
        exp_name = []
        for key, label in args_to_watch:
            if not hasattr(args, key):
                continue
            val = getattr(args, key)
            if type(val) == dict:
                val = '_'.join(f'{k}-{v}' for k, v in val.items())
            exp_name.append(f'{label}{val}')
        exp_name = '_'.join(exp_name)
        exp_name = exp_name.replace('/_', '/')
        exp_name = exp_name.replace('(', '').replace(')', '')
        exp_name = exp_name.replace(', ', '-')
        return exp_name
    return _fn

def lazy_fstring(template, args):
    ## https://stackoverflow.com/a/53671539
    return eval(f"f'{template}'")

class Parser(Tap):

    def save(self):
        fullpath = os.path.join(self.savepath, 'args.json')
        print(f'[ utils/setup ] Saved args to {fullpath}')
        super().save(fullpath, skip_unpicklable=True)

    def parse_args(self, experiment=None):
        args = super().parse_args(known_only=True)
        ## if not loading from a config script, skip the result of the setup
        if not hasattr(args, 'config'): return args
        args = self.read_config(args, experiment)
        self.add_extras(args)
        self.eval_fstrings(args)
        self.set_seed(args)
        self.get_commit(args)
        self.set_loadbase(args)
        self.generate_exp_name(args)
        self.mkdir(args)
        self.save_diff(args)
        return args

    def read_config(self, args, experiment):
        '''
            Load parameters from config file
        '''
        dataset = args.dataset.replace('-', '_')
        print(f'[ utils/setup ] Reading config: {args.config}:{dataset}')
        module = importlib.import_module(args.config)
        params = getattr(module, 'base')[experiment]

        if hasattr(module, dataset) and experiment in getattr(module, dataset):
            print(f'[ utils/setup ] Using overrides | config: {args.config} | dataset: {dataset}')
            overrides = getattr(module, dataset)[experiment]
            params.update(overrides)
        else:
            print(f'[ utils/setup ] Not using overrides | config: {args.config} | dataset: {dataset}')

        self._dict = {}
        for key, val in params.items():
            setattr(args, key, val)
            self._dict[key] = val

        return args

    def add_extras(self, args):
        '''
            Override config parameters with command-line arguments
        '''
        extras = args.extra_args
        if not len(extras):
            return

        print(f'[ utils/setup ] Found extras: {extras}')
        assert len(extras) % 2 == 0, f'Found odd number ({len(extras)}) of extras: {extras}'
        for i in range(0, len(extras), 2):
            key = extras[i].replace('--', '')
            val = extras[i+1]
            assert hasattr(args, key), f'[ utils/setup ] {key} not found in config: {args.config}'
            old_val = getattr(args, key)
            old_type = type(old_val)
            print(f'[ utils/setup ] Overriding config | {key} : {old_val} --> {val}')
            if val == 'None':
                val = None
            elif val == 'latest':
                val = 'latest'
            elif old_type in [bool, type(None)]:
                try:
                    val = eval(val)
                except:
                    print(f'[ utils/setup ] Warning: could not parse {val} (old: {old_val}, {old_type}), using str')
            else:
                val = old_type(val)
            setattr(args, key, val)
            self._dict[key] = val

    def eval_fstrings(self, args):
        for key, old in self._dict.items():
            if type(old) is str and old[:2] == 'f:':
                val = old.replace('{', '{args.').replace('f:', '')
                new = lazy_fstring(val, args)
                print(f'[ utils/setup ] Lazy fstring | {key} : {old} --> {new}')
                setattr(self, key, new)
                self._dict[key] = new

    def set_seed(self, args):
        if not hasattr(args, 'seed') or args.seed is None:
            return
        print(f'[ utils/setup ] Setting seed: {args.seed}')
        set_seed(args.seed)

    def set_loadbase(self, args):
        if hasattr(args, 'loadbase') and args.loadbase is None:
            print(f'[ utils/setup ] Setting loadbase: {args.logbase}')
            args.loadbase = args.logbase

    def generate_exp_name(self, args):
        if not 'exp_name' in dir(args):
            return
        exp_name = getattr(args, 'exp_name')
        if callable(exp_name):
            exp_name_string = exp_name(args)
            print(f'[ utils/setup ] Setting exp_name to: {exp_name_string}')
            setattr(args, 'exp_name', exp_name_string)
            self._dict['exp_name'] = exp_name_string

    def mkdir(self, args):
        if 'logbase' in dir(args) and 'dataset' in dir(args) and 'exp_name' in dir(args):
            args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name)
            self._dict['savepath'] = args.savepath
            if 'suffix' in dir(args):
                args.savepath = os.path.join(args.savepath, args.suffix)
            if mkdir(args.savepath):
                print(f'[ utils/setup ] Made savepath: {args.savepath}')
            self.save()

    def get_commit(self, args):
        args.commit = get_git_rev()

    def save_diff(self, args):
        try:
            save_git_diff(os.path.join(args.savepath, 'diff.txt'))
        except:
            print('[ utils/setup ] WARNING: did not save git diff')


def set_normalizer_bounds(cfg, dataset, observation_dim: int, action_dim: int) -> None:
    """
    Set the observation and action bounds in config from dataset normalizers.
    
    This function extracts the min/max normalization bounds from the dataset's 
    normalizers and stores them in the config object for later use.
    
    Args:
        cfg: Config object with obs_max, obs_min, act_max, act_min lists/arrays
             Type: FlowMatchingEvaluationConfig or similar config object
        dataset: Dataset object containing normalizer information
                Type: GoalDataset or similar with normalizer attribute
        observation_dim: Dimension of observation space
                       Type: int
        action_dim: Dimension of action space
                   Type: int
    
    Returns:
        None (modifies cfg in place)
        
    Data format:
        - cfg.obs_max: list/array of size (observation_dim,)
        - cfg.obs_min: list/array of size (observation_dim,)
        - cfg.act_max: list/array of size (action_dim,)
        - cfg.act_min: list/array of size (action_dim,)
    """
    for i in range(observation_dim):
        cfg.obs_max[i] = dataset.normalizer.normalizers["observations"].maxs[i]
        cfg.obs_min[i] = dataset.normalizer.normalizers["observations"].mins[i]
        
    for i in range(action_dim):
        cfg.act_max[i] = dataset.normalizer.normalizers["actions"].maxs[i]
        cfg.act_min[i] = dataset.normalizer.normalizers["actions"].mins[i]


def load_dynamics_model(cfg: FlowMatchingEvaluationConfig, observation_dim: int, action_dim: int):
    """
    Load a trained dynamics model and set it to evaluation mode.
    
    This function initializes a dynamics model, loads the pre-trained weights 
    from checkpoint, and sets the model to evaluation mode. Currently supports 
    'smooth' dynamics models.
    
    Args:
        cfg: Config for evaluation
        observation_dim: Dimension of observation/state space
                       Type: int
        action_dim: Dimension of action space
                   Type: int
    
    Returns:
        dynamics_model: Loaded and initialized dynamics model in eval mode
                       Type: torch.nn.Module (specifically Maze2dNNDynamicsModel)
    """
    # get dynamic
    if cfg.NN_folder == 'smooth':
        from gflower.models_dynamic.maze2d_dynamics import Maze2dNNDynamicsModel
        FD = Maze2dNNDynamicsModel(
            state_dim=observation_dim, 
            action_dim=action_dim, 
            hidden_dim=cfg.hidden_dim
        ).to(cfg.device)
    else:
        raise NotImplementedError("only smooth dynamics model is supported for now")
        
    # Determine checkpoint filename
    if cfg.IsEma is True:
        file_name = f'model_ema_esemble_{cfg.NN_esemble_idx}_{cfg.NN_model_idx}.pth'
    else:
        file_name = f'model_esemble_{cfg.NN_esemble_idx}_{cfg.NN_model_idx}.pth'
    
    # Load checkpoint
    FD_checkpoint = torch.load(os.path.join(
        cfg.log_folder, cfg.env, "model_dynamics", cfg.NN_folder, file_name
    ))
    FD.load_state_dict(FD_checkpoint)
    FD.eval()
    
    return FD


def initialize_flow_policy(cfg: FlowMatchingEvaluationConfig, normalizer, observation_dim: int, action_dim: int, FD):
    """
    Initialize and create the complete flow policy with all necessary models.
    
    This function orchestrates the creation of flow_transformer, value_model, guide_model,
    and flow_policy. It loads pre-trained checkpoints and sets models to appropriate modes.
    
    Args:
        cfg: Configuration object for evaluation
        
        normalizer: Dataset normalizer object for action/observation normalization
                   Type: object with normalization parameters
        observation_dim: Dimension of observation/state space
                        Type: int
        action_dim: Dimension of action space
                   Type: int
        FD: Pre-loaded dynamics model
           Type: torch.nn.Module (Maze2dNNDynamicsModel)
    
    Returns:
        flow_policy: Initialized FlowPolicy object ready for evaluation
                    Type: FlowPolicy
    """
    from gflower.models_flow.transformer import TransformerFlow
    from gflower.models_value.transformer import Transformer as ValueTransformer
    from gflower.models_flow.flow_policy import FlowPolicy
    
    # Initialize flow transformer (policy network)
    flow_transformer = TransformerFlow(
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
    
    flow_transformer.load_state_dict(torch.load(os.path.join(
        cfg.log_folder, cfg.env, 'flow', f"{cfg.flow_exp_name}_Hor_{cfg.horizon}", f'model_ema_{cfg.flow_cp}.pth'
    )))
    
    # Initialize value model (conditional based on guidance_method)
    if cfg.guidance_method not in ['no']:
        value_model = ValueTransformer(
            input_dim=observation_dim + action_dim,
            output_dim=1,
            model_dim=cfg.value_transformer_config.model_dim,
            num_heads=cfg.value_transformer_config.num_heads,
            num_layers=cfg.value_transformer_config.num_layers,
            dropout=cfg.value_transformer_config.dropout,
        ).to(cfg.device)
        value_model.load_state_dict(torch.load(os.path.join(
            cfg.log_folder, cfg.env, 'value', f"{cfg.value_exp_name}_Hor_{cfg.horizon}", f'model_{cfg.value_cp}.pth'
        )))
    else:
        value_model = None
    
    # Initialize guidance model (conditional based on guidance_method)
    if cfg.guidance_method == 'guidance_matching':
        guide_model = TransformerFlow(
            seq_len=cfg.horizon,
            in_channels=observation_dim + action_dim,
            out_channels=(observation_dim + action_dim) if cfg.guide_matching_type != 'grad_z' else 1,
            hidden_size=cfg.guide_model_transformer_config.hidden_size,
            depth=cfg.guide_model_transformer_config.depth,
            num_heads=cfg.guide_model_transformer_config.num_heads,
            mlp_ratio=cfg.guide_model_transformer_config.mlp_ratio,
            x_emb_proj=cfg.guide_model_transformer_config.x_emb_proj,
            x_emb_proj_conv_k=cfg.guide_model_transformer_config.x_emb_proj_conv_k,
        ).to(cfg.device)
        
        if cfg.guide_matching_type != 'grad_z':
            guide_model.load_state_dict(torch.load(os.path.join(
                cfg.log_folder, cfg.env, 'guidance', cfg.guide_model_exp_name, f'model_{cfg.guide_matching_type}_{cfg.guide_model_cp}.pth'
            )))
        else:
            guide_model.load_state_dict(torch.load(os.path.join(
                cfg.log_folder, cfg.env, 'guidance', cfg.guide_model_exp_name, f'model_z_{cfg.guide_model_cp}.pth'
            )))
    else:
        guide_model = None
    
    # Create flow policy combining all models
    flow_policy = FlowPolicy(
        flow_model=flow_transformer,
        value_model=value_model,
        guide_model=guide_model,
        normalizer=normalizer,
        action_dim=action_dim,
        state_dim=observation_dim,
        horizon=cfg.horizon,
        cfg=cfg,
        FD_model=FD,
    )
    
    return flow_policy


def initialize_renderer(cfg, observation_dim: int):
    """
    Initialize and configure a maze renderer for visualization.
    
    This function creates a Maze2dRenderer instance if rendering is enabled,
    sets up obstacle information, and returns the renderer or None.
    
    Args:
        cfg: Configuration object containing rendering settings
        observation_dim: Dimension of observation/state space (used for renderer initialization)
                        Type: int
    
    Returns:
        renderer: Initialized Maze2dRenderer with obstacles configured, or None
                 Type: Maze2dRenderer or None
    """
    from gflower.utils.rendering import Maze2dRenderer
    
    if cfg.IsRender:
        renderer = Maze2dRenderer(cfg.env, observation_dim)
        renderer.set_obstacle(cfg.obs_center, cfg.obs_radius)
    else:
        renderer = None
    
    return renderer


def save_metric_per_iteration(cfg: FlowMatchingEvaluationConfig, metric_value, metric_name: str, iteration_idx: int, value_label: str = None) -> None:
    """
    Save individual metric result for a specific iteration to a per-iteration file.
    
    This function writes a single metric value to a separate file for each iteration.
    Can be used for recording individual scores, times, or any other scalar metrics.
    
    Args:
        cfg: Configuration object containing path information
        
        metric_value: The metric value to save (single numeric value)
                     Type: float, int, or any numeric type
        metric_name: Name of the metric type (e.g., 'score', 'time')
                    Type: str
        iteration_idx: Index/number of the iteration or run
                      Type: int
        value_label: Optional label prefix for output (e.g., 'score:', 'comp_time:')
                    Type: str, optional (defaults to '{metric_name}:' if None)
    
    Returns:
        None
    """
    if value_label is None:
        value_label = f"{metric_name}:"
    
    filepath = os.path.join(cfg.log_folder, cfg.env, 'eval', cfg.exp_name, f'{metric_name}_{iteration_idx}.txt')
    with open(filepath, 'a+') as f:
        f.write(f"{value_label} {metric_value}\n")


def save_metrics_aggregated(cfg: FlowMatchingEvaluationConfig, metrics_list, metric_name: str, value_label: str = None) -> None:
    """
    Save aggregated metric statistics (mean, std, and full list) to file.
    
    This function computes mean and standard deviation from a list of metrics
    and saves both the aggregated statistics and the complete metric list to a single file.
    Can be used for recording aggregate scores, times, or any other metrics.
    
    Args:
        cfg: Configuration object containing path information

        metrics_list: List of individual metric values to aggregate
                     Type: list or np.ndarray of floats/ints
        metric_name: Name of the metric type (e.g., 'score', 'time')
                    Type: str
        value_label: Optional label prefix for output (e.g., 'score', 'time')
                    Type: str, optional (defaults to metric_name if None)
    
    Returns:
        None
        
    """
    if value_label is None:
        value_label = metric_name
    
    metrics = np.array(metrics_list)
    metric_mean = metrics.mean()
    metric_std = metrics.std()
    
    filepath = os.path.join(cfg.log_folder, cfg.env, 'eval', cfg.exp_name, f'{metric_name}_results.txt')
    with open(filepath, 'a+') as f:
        f.write(f"{value_label}: {metric_mean} +- {metric_std}\n")
        f.write(f"{metric_name}s: {metrics_list}\n")
