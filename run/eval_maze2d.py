import json
from run.utils import deterministic, save_config, set_cuda_visible_device
import numpy as np
import tqdm
from os.path import join
import os, torch, sys, tyro
from gflower.config.flow_matching import FlowMatchingEvaluationConfig
from gflower.utils.rendering import Maze2dRenderer
from gflower.datasets.sequence import GoalDataset
from gflower.models_flow.transformer import TransformerFlow
from gflower.models_value.transformer import Transformer as ValueTransformer
from gflower.models_flow.flow_policy import FlowPolicy
from gflower.utils.debug import Create_violation_title, Summary_violationfile, GetExpName, Record_violation_into_file
from gflower.utils.setup import set_normalizer_bounds, load_dynamics_model, initialize_flow_policy, initialize_renderer, save_metric_per_iteration, save_metrics_aggregated

def evaluate(cfg: FlowMatchingEvaluationConfig):
    dataset = GoalDataset(
        env=cfg.env,
        horizon=cfg.horizon,
        normalizer=cfg.normalizer,
        preprocess_fns=cfg.preprocess_fns,
        max_path_length=cfg.max_path_length,
        seed=cfg.seed,
    )
    
    # Set normalization bounds from dataset
    set_normalizer_bounds(cfg, dataset, dataset.observation_dim, dataset.action_dim)
    
    # Load dynamics model
    FD = load_dynamics_model(cfg, dataset.observation_dim, dataset.action_dim)
    
    # Initialize flow policy with all models
    flow_policy = initialize_flow_policy(cfg, dataset.normalizer, dataset.observation_dim, dataset.action_dim, FD)
    
    # Initialize renderer
    renderer = initialize_renderer(cfg, dataset.observation_dim)
    
    scores_ls = []
    rollout_ls = []
    action_ls =[]
    time_ls = []
    for i in range(cfg.random_repeat):
        # set record file for this evaluation
        FinalPolicyfilename = f"{cfg.final_polciy_file_name}_{i}.txt"
        FinalPolicyfullfilepath = os.path.join(log_subfolder,FinalPolicyfilename)
        flow_policy.set_record_file(FinalPolicyfullfilepath)
        cfg.nEvaluaIdx = i
        
        # run environment with flow policy
        score, comp_time, rollout, action_roollout = run_env(dataset.env, flow_policy, cfg, log_subfolder, i, renderer)
        
        # record results
        scores_ls.append(score)
        time_ls.append(comp_time)
        rollout_ls.append(np.array(rollout))
        action_ls.append(np.array(action_roollout))
        
        # dumped results into file
        print("---------- evaluation finished, summary record file------------------")
        Summary_violationfile(FinalPolicyfullfilepath)
        
        os.makedirs(cfg.result_folder, exist_ok= True)
        # save state, action, time, and score for each rollout
        np.save(os.path.join(cfg.result_folder, f'rollout{i}.npy'), rollout_ls[i])
        np.save(os.path.join(cfg.result_folder, f'rollout{i}_act.npy'), action_ls[i])
        save_metric_per_iteration(cfg, scores_ls[i], 'score', i)
        save_metric_per_iteration(cfg, time_ls[i], 'time', i, value_label='comp_time:')
        
        
    # compute and save statistic metrics
    save_metrics_aggregated(cfg, scores_ls, 'score')
    save_metrics_aggregated(cfg, time_ls, 'time')
        


def run_env(env, policy: FlowPolicy, cfg: FlowMatchingEvaluationConfig, log_subfolder:str, 
            env_time:int, renderer=None):
    observation = env.reset()
    
    if cfg.env == 'maze2d-large-v1':
        if env_time == 0 or env_time == 1:
            observation = np.array([1.+np.random.uniform(-0.2, 0.2),1.+np.random.uniform(-0.2, 2),0.,0.])
            env.set_state(observation[:2], observation[2:])
        if env_time == 4:
            observation = np.array([3.+np.random.uniform(-0.2, 0.2),1.+np.random.uniform(-0.2, 2),0.,0.])
            env.set_state(observation[:2], observation[2:])
    
    target = env._target
    conditions = {
        cfg.horizon - 1: np.array([*target, 0, 0]),
    }
    rollout = [observation.copy()] # for rendering
    act_rollout = []
    total_reward = 0
        
    terminal = False
    # Create tqdm progress bar with reward and score as postfix
    pbar = tqdm.tqdm(range(cfg.max_episode_length), desc='Episode')
    # dumped files for this rollout
    plan_path = join(log_subfolder,"maze_render", f'{env_time}th_plan.png')
    plan_obs_path = join(log_subfolder, f'rollout{env_time}_planned_obs.npy')
    plan_act_path = join(log_subfolder, f'rollout{env_time}_planned_act.npy')
    control_path = join(log_subfolder, "maze_render", f'{env_time}th_rollout.png')
    
    # planning with flow policy, and then control with PD controller
    for t in pbar:
        state = env.state_vector().copy()
        
        if t == 0:
            conditions[0] = observation
            
            # samples: full plan result, including obs and act, with shape [hor, dim]
            action, samples, comp_time = policy(conditions, batch_size=cfg.batch_size) #samples.obs, samples.act: [hor,dim]
            
            sequence = samples.observations #[hor, obs_dim]
            
            os.makedirs(join(log_subfolder,"maze_render"),exist_ok=True)
            
        # waypoint following with PD control
        if t < sequence.shape[0] - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            
        p_error = next_waypoint[:2] - state[:2]
        v_error = (next_waypoint[2:] - state[2:])
        action = p_error + v_error

        next_observation, reward, _terminated, _ = env.step(action) # TODO: make compatible with gymnasium
        total_reward += reward
        score = env.get_normalized_score(total_reward)
        
        # show position
        xy = next_observation[:2]
        goal = env.unwrapped._target
        print(
            f'maze | pos: {xy} | goal: {goal}'
        )

        # Update progress bar postfix with reward and score
        pbar.set_postfix({'reward': f'{reward:.3f}', 'score': f'{score:.3f}'})
        
        # record and update
        rollout.append(next_observation.copy())
        act_rollout.append(action.copy())
        
        mse = np.sqrt(np.mean((next_observation[0:2]-target) ** 2))
        if mse <= 0.001:
            terminal = True
        
        # render
        if t % cfg.render_freq == 0 or terminal:
            if t == 0: 
                renderer.composite(plan_path, samples.observations[None], ncol=1)
                np.save(plan_obs_path, samples.observations)
                np.save(plan_act_path, samples.actions)
            ## save rollout thus far
            renderer.composite(control_path, np.array(rollout)[None], ncol=1)
        
        if terminal:
            break
        
        observation = next_observation
    
    return score, comp_time, rollout, act_rollout




if __name__ == '__main__':
    
    cfg = tyro.cli(FlowMatchingEvaluationConfig)
    cfg.exp_name = GetExpName(cfg)
    set_cuda_visible_device(cfg)
    deterministic(cfg.seed) # seed everything
    log_subfolder = os.path.join(cfg.log_folder, cfg.env, 'eval', cfg.exp_name)
    cfg.result_folder = log_subfolder
    
    if cfg.Is_Float64:
        torch.set_default_dtype(torch.float64)
    
    # create debug file title
    for i in range(cfg.random_repeat):
        final_policy_name = f"{cfg.final_polciy_file_name}_{i}.txt"
        if Create_violation_title(log_subfolder, final_policy_name) is False:
            print("file is there, make sure that the file is saved and move")
            print(log_subfolder)
            sys.exit()
            
    print("----------create_record_file, now start evaluation------------------")
    
    save_config(cfg, log_subfolder)
    evaluate(cfg)
    