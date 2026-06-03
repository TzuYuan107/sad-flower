import torch
import os, sys
import statistics
from gflower.config.flow_matching import FlowMatchingEvaluationConfig
from typing import List
import math

def GetExpName(cfg:FlowMatchingEvaluationConfig) -> str:
    if cfg.constraint_strategy == "No":
        if cfg.IsEma is True:
            ema_name = "ema"
        else:
            ema_name = "normal"
        cfg.exp_name = f"No_H{cfg.horizon}/{cfg.NN_folder}/odeint_NN{ema_name}_ode{cfg.ode_t_steps}"
    elif cfg.constraint_strategy == "Reject":
        if cfg.IsEma is True:
            ema_name = "ema"
        else:
            ema_name = "normal"
        cfg.exp_name = f"Reject_H{cfg.horizon}/{cfg.NN_folder}/odeint_NN{ema_name}_ode{cfg.ode_t_steps}"
    else:
        if cfg.constraint_strategy == "QP_PT_ACSCDC":
            if cfg.ACSCDCFlag == 0:
                strategy_folder="ACSCDC_AllInOne"
            elif cfg.ACSCDCFlag == 1:
                strategy_folder=f"ExpJac_ACSCDC_AllInOne_ALim{cfg.act_pos_lim}"
            else:
                return NotImplementedError
        else:
            print("wrong constraint_strategy setting")
            sys.exit()
        
        if abs(cfg.Stop_V)>0.0001:
            strategy_folder = strategy_folder + f"StopV_{cfg.Stop_V}"
        
        if cfg.IsEma is True:
            ema_name = "ema"
        else:
            ema_name = "normal"
        
        if cfg.Is_Float64:
            cfg.exp_name = f"{strategy_folder}_H{cfg.horizon}/{cfg.NN_folder}/{cfg.CBF_c[0]}_{cfg.CBF_c[1]}_{cfg.CLF_c}_CBFt_{cfg.PT_CBF_min}_CLF_t_{cfg.PT_CLF_min}_NN{ema_name}_ode{cfg.ode_t_steps}_start{cfg.start_time}_{cfg.ode_solver}_dsarobust{cfg.d_robust}_{cfg.s_robust}_{cfg.a_robust}_Float64"
        else:
            cfg.exp_name = f"{strategy_folder}_H{cfg.horizon}/{cfg.NN_folder}/{cfg.CBF_c[0]}_{cfg.CBF_c[1]}_{cfg.CLF_c}_CBFt_{cfg.PT_CBF_min}_CLF_t_{cfg.PT_CLF_min}_NN{ema_name}_ode{cfg.ode_t_steps}_start{cfg.start_time}_{cfg.ode_solver}_dsarobust{cfg.d_robust}_{cfg.s_robust}_{cfg.a_robust}"
    
    return cfg.exp_name

def Create_violation_title(recordfilepath:str, recordfilename:str) ->bool:
    fullpath = os.path.join(recordfilepath, recordfilename)
    if os.path.isfile(fullpath):
        return False
    
    os.makedirs(recordfilepath, exist_ok=True)
    with open(fullpath, "w") as f:
        header_format = (
            "{:<10}| {:<10}| {:<10}| {:<15}| {:<15}| {:<15}| "
            "{:<10}| {:<10}| {:<10}| {:<15}| {:<15}| {:<15}|{:<15}\n"
        )
        f.write(header_format.format(
            "s_vio_num", "a_vio_num", "d_vio_num",
            "s_vio_num_thr", "a_vio_num_thr", "d_vio_num_thr",
            "max_s_vio", "max_a_vio", "max_d_vio",
            "max_s_vio_idx","max_a_vio_idx", "max_d_vio_idx", "mean_d_vio"
        ))
        
    return True

def Record_violation_into_file(sol:torch.tensor, action_dim, nMap:int,
                               obs_center_norm:List[torch.tensor], obs_radius_norm:List[torch.tensor], limit_AC,
                               FD_model,
                               record_file_path:str):
    idx_state_0 = action_dim
    batch, hor, dim = sol.shape
    n_obs = len(obs_center_norm)
    h_x = torch.zeros(batch, hor, n_obs)
    if nMap == 1: #Umaze
        h_x[:,:,0] = (((sol[:,:,2:4] - obs_center_norm[0])/obs_radius_norm[0])**4).sum(dim=-1) - 1
        h_x[:,:,1] = 1 - (((sol[:,:,2:4] - obs_center_norm[1])/obs_radius_norm[1])**4).sum(dim=-1)
    else:
        for i in range(n_obs):
            h_x[:,:,i] = (((sol[:,:,2:4] - obs_center_norm[i])/obs_radius_norm[i])**2).sum(dim=-1) - 1 #[batch,hor,n_obs]
    b_x_pos = (limit_AC[1] - sol[:,:,:idx_state_0]) #[batch,hor,3]
    b_x_neg = (sol[:,:,:idx_state_0] - limit_AC[0]) #[batch,hor,3]
    b_x = torch.cat([b_x_pos, b_x_neg], dim=2) #[batch,hor,6]
    
    cur_state = sol[:,:-1,idx_state_0:]
    cur_act = sol[:,:-1,:idx_state_0]
    next_state = sol[:,1:,idx_state_0:]
    d_x = (0.5*(next_state - FD_model(cur_state, cur_act))**2).sum(dim=2, keepdim=True) #[batch,hor-1,1]
    d_x_mean = d_x.mean().item()
    # voilation number
    mask_state = h_x < 0 #h(x)<0
    mask_act =  b_x < 0 #b(x)<0
    mask_d = d_x > 0.0001 
    num_violate_state = mask_state.sum().item()
    num_violate_act = mask_act.sum().item()
    num_violate_d = mask_d.sum().item()
    
    # voilation number with threshold
    thres = 0.005
    mask_state_thr = h_x < -thres #h(x)<0
    mask_act_thr =  b_x < -thres #b(x)<0
    mask_d_thr = d_x > thres
    
    num_violate_state_thr = mask_state_thr.sum().item()
    num_violate_act_thr = mask_act_thr.sum().item()
    num_violate_d_thr = mask_d_thr.sum().item()
    
    # max violation, idx
    max_violation_state, max_violation_state_idx = h_x.view(-1).min(0)
    max_violation_act, max_violation_act_idx = b_x.view(-1).min(0)
    max_violation_d, max_violation_d_idx = d_x.view(-1).max(0)
    
    batch_size, hor, dim = h_x.shape
    i_state = max_violation_state_idx // (hor * dim)
    j_state = (max_violation_state_idx % (hor * dim)) // dim
    k_state = max_violation_state_idx % dim
    
    batch_size, hor, dim = b_x.shape
    i_action = max_violation_act_idx // (hor * dim)
    j_action = (max_violation_act_idx % (hor * dim)) // dim
    k_action = max_violation_act_idx % dim
    
    batch_size, hor, dim = d_x.shape
    i_d = max_violation_d_idx // (hor * dim)
    j_d = (max_violation_d_idx % (hor * dim)) // dim
    k_d = max_violation_d_idx % dim
    
    
    with open(record_file_path, "a") as f:
        data_format = (
            "{:<10}| {:<10}| {:<10}| {:<15}| {:<15}| {:<15}| "
            "{:<10.4f}| {:<10.4f}| {:<10.4f}| {:<15}| {:<15}| {:<15}| {:<15}\n"
        )
        f.write(data_format.format(
        num_violate_state,
        num_violate_act,
        num_violate_d,
        num_violate_state_thr,
        num_violate_act_thr,
        num_violate_d_thr,
        max_violation_state,
        max_violation_act,
        max_violation_d,
        f"{tuple([i_state.item(), j_state.item(), k_state.item()])}",
        f"{tuple([i_action.item(), j_action.item(), k_action.item()])}",
        f"{tuple([i_d.item(), j_d.item(), k_d.item()])}",
        d_x_mean
        ))

def Summary_violationfile(record_file_path:str):
    # Read the file and extract data lines
    with open(record_file_path, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip()
    data_lines = [line.strip() for line in lines[1:] if line.strip()]

    # Extract columns
    columns = [col.strip() for col in header.split('|')]
    records = [
        [item.strip() for item in line.split('|')]
        for line in data_lines
    ]

    # Convert to dictionary list for processing
    numeric_fields = [
        's_vio_num', 'a_vio_num', 'd_vio_num',
        's_vio_num_thr', 'a_vio_num_thr', 'd_vio_num_thr',
        'max_s_vio', 'max_a_vio', 'max_d_vio'
    ]
    idx_fields = ['max_s_vio_idx', 'max_a_vio_idx', 'max_d_vio_idx']

    col_indices = {col: idx for idx, col in enumerate(columns)}

    # Compute max and mean
    def get_float(col):
        idx = col_indices[col]
        return [float(row[idx]) for row in records]

    summary = {
        's_vio_num': max(get_float('s_vio_num')),
        'a_vio_num': max(get_float('a_vio_num')),
        'd_vio_num': max(get_float('d_vio_num')),
        's_vio_num_thr': max(get_float('s_vio_num_thr')),
        'a_vio_num_thr': max(get_float('a_vio_num_thr')),
        'd_vio_num_thr': max(get_float('d_vio_num_thr')),
        'max_s_vio': min(get_float('max_s_vio')),
        'max_a_vio': min(get_float('max_a_vio')),
        'max_d_vio': max(get_float('max_d_vio')),
        'max_mean_d_vio': max(get_float('mean_d_vio')),
        'mean_mean_d_vio': statistics.mean(get_float('mean_d_vio'))
    }

    # Construct formatted row
    row = (
        "{:<10}| {:<10}| {:<10}| {:<15}| {:<15}| {:<15}| "
        "{:<10}| {:<10}| {:<10}| {:<15}| {:<15}| {:<15}|{:<15}\n"
    ).format(
        summary['s_vio_num'], summary['a_vio_num'], summary['d_vio_num'],
        summary['s_vio_num_thr'], summary['a_vio_num_thr'], summary['d_vio_num_thr'],
        summary['max_s_vio'], summary['max_a_vio'], summary['max_d_vio'],
        '', '', '',  # blank for *_idx
        f"{summary['max_mean_d_vio']:.4f}, {summary['mean_mean_d_vio']:.4f}"
    )

    # Append to file
    with open(record_file_path, 'a') as f:
        f.write(row)

    print("Summary row appended successfully.")

def GetMinSCAC_MaxDC(record_file_path:str):
    # Read the file and extract data lines
    with open(record_file_path, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip()
    data_lines = [line.strip() for line in lines[1:] if line.strip()]

    # Extract columns
    columns = [col.strip() for col in header.split('|')]
    records = [
        [item.strip() for item in line.split('|')]
        for line in data_lines
    ]
    
    SC_Min_idx = 6
    AC_Min_idx = 7
    DC_Mean_idx = 12
    MinSC = float(records[-1][SC_Min_idx])
    MinAC = float(records[-1][AC_Min_idx])
    MeanDC = float(records[-1][DC_Mean_idx].split()[1])
    
    return MinSC, MinAC, MeanDC

def save_main_result(SC_min, AC_min, SC_mean, AC_mean, SC_std, AC_std, DC_mean, DC_std, record_path):
    """
    Records scalar values to a text file with a two-row, three-column format.
    
    Args:
        SC_min (float): The minimum state cost.
        AC_min (float): The minimum action cost.
        DC_mean (float): The mean of the dynamic cost.
        DC_std (float): The standard deviation of the dynamic cost.
        record_path (str): The directory path to save the file.
    """
    # Construct the full file path for the output file
    output_filepath = f"{record_path}/record_main_result.txt"

    # Use a try-except block for robust file handling
    try:
        with open(output_filepath, 'w') as f:
            # Write the header row
            header_row = f"{'SC_min':<10}{'AC_min':<10}{'SC_mean':<25}{'AC_mean':<25}{'DC':<25}\n"
            f.write(header_row)

            # Write the data row with aligned values
            data_row = f"{SC_min:<10.3f}{AC_min:<10.3f}  {SC_mean:10.3f}+-{SC_std:.3f}  {AC_mean:10.3f}+-{AC_std:.3f}  {DC_mean:.3f}+-{DC_std:.3f}\n"
            f.write(data_row)
        print(f"Successfully saved results to {output_filepath}")
    except Exception as e:
        print(f"Error: Failed to save results to file. Details: {e}")

    
def GetMapIndex(env_name:str) -> int:
    if env_name == "maze2d-open-dense-v0":
       return 0
    elif env_name == "maze2d-umaze-v1":
        return 1
    elif env_name == "maze2d-large-v1":
        return 2
    else:
        print("wrong map")
        sys.exit()
        
def ConstructExpForwardModel(cfg:FlowMatchingEvaluationConfig):
    """
    return [obs_dim, obs_dim], [obs_dim, act_dim]
    """
    alpha = cfg.gear_ratio/cfg.point_mass
    beta = cfg.viscious/cfg.point_mass
    dt = cfg.sim_time
    if abs(cfg.viscious)<0.00001: # no damping in joint
        expJac_s = torch.tensor([[  1., 0., dt, 0.],
                                [  0., 1., 0., dt],
                                [  0., 0., 1., 0.],
                                [  0., 0., 0., 1.]], device=cfg.device)
        
        expJac_a = torch.tensor([[  0.5*alpha*(dt**2),  0.],
                                [  0.,                 0.5*alpha*(dt**2)],
                                [  alpha*dt,           0.],
                                [  0.,                 alpha*dt]], device=cfg.device)
    else:
        exp_term = math.exp(-beta*dt)
        exp_overbeta = (1-exp_term)/beta
        
        expJac_s = torch.tensor([[  1., 0., exp_overbeta, 0.],
                                [  0., 1., 0., exp_overbeta],
                                [  0., 0., exp_term, 0.],
                                [  0., 0., 0., exp_term]], device=cfg.device)
        
        expJac_a = torch.tensor([[  alpha*(dt/beta - (1-exp_term)/(beta ** 2)),  0.],
                                [  0.,                 alpha*(dt/beta - (1-exp_term)/(beta ** 2))],
                                [  alpha*exp_overbeta,           0.],
                                [  0.,                 alpha*exp_overbeta]], device=cfg.device)
    
    return expJac_s, expJac_a


def ConstructExpForwardModelNormed(cfg:FlowMatchingEvaluationConfig, nBatch:int, nHor:int):
    A, B = ConstructExpForwardModel(cfg)
    
    if cfg.Is_Float64:
        max_s = torch.tensor(cfg.obs_max, dtype=torch.float64).to(cfg.device)
        min_s = torch.tensor(cfg.obs_min, dtype=torch.float64).to(cfg.device)
        max_a = torch.tensor(cfg.act_max, dtype=torch.float64).to(cfg.device)
        min_a = torch.tensor(cfg.act_min, dtype=torch.float64).to(cfg.device)
    else:
        max_s = torch.tensor(cfg.obs_max, dtype=torch.float32).to(cfg.device)
        min_s = torch.tensor(cfg.obs_min, dtype=torch.float32).to(cfg.device)
        max_a = torch.tensor(cfg.act_max, dtype=torch.float32).to(cfg.device)
        min_a = torch.tensor(cfg.act_min, dtype=torch.float32).to(cfg.device)
    
    normed_s = 2/(max_s-min_s) #[obs_dim]
    Sigma_s = torch.diag(normed_s).to(cfg.device) #[obs_dim, obs_dim]
    Sigma_s_inv = torch.linalg.inv(Sigma_s)
    Os = -Sigma_s @ min_s - 1
    
    normed_a = 2/(max_a-min_a) #[act_dim]
    Sigma_a = torch.diag(normed_a).to(cfg.device) #[act_dim, act_dim]
    Sigma_a_inv = torch.linalg.inv(Sigma_a)
    Oa = -Sigma_a @ min_a - 1
    
    # get normed dynamics s_normed_{k+1} = A_normed*s_normed_{k}+ B_normed*a_normed_{k}+ C_normed
    A_normed = Sigma_s @ A @ Sigma_s_inv # [obs_dim, obs_dim]
    B_normed = Sigma_s @ B @ Sigma_a_inv # [obs_dim, act_dim]
    C_normed = -Sigma_s @ A @ Sigma_s_inv @ Os - Sigma_s @ B @ Sigma_a_inv @ Oa + Os #[obs_dim]
    
    A_normed = A_normed.unsqueeze(0).unsqueeze(0).expand(nBatch, nHor, -1, -1) # [B, H, obs_dim, obs_dim]
    B_normed = B_normed.unsqueeze(0).unsqueeze(0).expand(nBatch, nHor, -1, -1) # [B, H, obs_dim, act_dim]
    C_normed = C_normed.unsqueeze(0).unsqueeze(0).expand(nBatch, nHor, -1) # [B, H, obs_dim]
    
    return A_normed, B_normed, C_normed

# Custom collate function to ensure double precision
def collate_fn(batch):
    # Assuming batch is a list of tuples/tensors
    if isinstance(batch[0], (tuple, list)):
        return tuple(torch.stack([item[i].double() for item in batch]) for i in range(len(batch[0])))
    else:
        return torch.stack([item.double() for item in batch])
    
    
