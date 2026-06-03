from collections import namedtuple
import math
import numpy as np
from torch import nn
import torch
from gflower.config.flow_matching import FlowMatchingEvaluationConfig
from gflower import utils
from gflower.models_flow.flow_matcher import apply_conditioning, apply_conditioning_from_conditioned_x
from gflower.models_flow.optimal_transport import OTPlanSampler
from gflower.utils.arrays import to_torch
from gflower.CBFCLF_tool.constraint_strategy import *
from torchdiffeq import odeint
from gflower.utils.debug import Record_violation_into_file, GetMapIndex, ConstructExpForwardModelNormed
from typing import List
import time

Trajectories = namedtuple('Trajectories', 'actions observations values')

def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

class ConditionedODESolver(nn.Module):
    def __init__(self, model, conditions, action_dim, guide_fn=None, ode_method='euler'):
        super().__init__()
        self.model = model
        self.conditions = conditions
        self.guide_fn = guide_fn
        self.action_dim = action_dim
        assert ode_method in ['euler'], "Only Euler is supported for now"

    def set_condition(self, conditions):
        self.conditions = conditions
    
    def forward(self, x, t_span, *args, **kwargs):
        """
        Args:
            x (torch.tensor) [batch, horizon, act_dim+state_dim]
            t_span (torch.tensor) [ode_time_step]
        """
        assert len(t_span) > 1, "t_span must have at least 2 elements"
        x0 = x.clone()
        dt = t_span[1] - t_span[0]
        for t in t_span:
            if self.guide_fn is None:
                # model forward pass
                dx_dt = self.model(x, t)
            # add gradient guidance
            else:
                x = x.requires_grad_()
                dx_dt = self.model(x, t)
                dx_dt = dx_dt + self.guide_fn(x, t, dx_dt, self.model)
            # fill in the condition, which means apply 0 in dx_dt
            dx_dt = apply_conditioning_from_conditioned_x(
                dx_dt, torch.zeros_like(x), self.conditions, self.action_dim
            )
            x = x + dx_dt * dt
            x = x.detach()
        return x

    
class CBFCLFODESolver(nn.Module):
    def __init__(self, model, action_dim, state_dim, hor,
                 start_time, prescribe_time, 
                 cfg:FlowMatchingEvaluationConfig,
                 dt, nMap:int,
                 guide_fn=None, FD_model=None, 
                 atol=1e-9, rtol=1e-7, ode_method='euler', ACSCDCFlag=1,
                 ):
        super().__init__()
        self.model = model
        self.guide_fn = guide_fn
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hor = hor
        self.ode_method = ode_method
        self.atol = atol
        self.rtol = rtol
        self.b_strategy = False
        self.conditions = None
        self.stategy_map : dict = { 'No': None,
                                    'QP_PT_ACSCDC': QP_PT_ACSCDC
                                    }
        
        self.start_time = start_time
        self.prescibed_time = prescribe_time
        self.FD_model = FD_model
        self.ACSCDCFlag = ACSCDCFlag
        self.cfg = cfg
        self.dt = dt
        self.nMap = nMap
            
        
        self.A_normed, self.B_normed, self.C_normed = ConstructExpForwardModelNormed(cfg, cfg.ss_batch * cfg.batch_size, cfg.horizon-1)
        
        self.b_expModel = False
        if cfg.ACSCDCFlag == 1:
            self.b_expModel = True
                   
    def set_strategy(self, option_key:str):
        strategyClass = self.stategy_map.get(option_key)
        
        if strategyClass is None:
            self.b_strategy = False
        else:
            if not strategyClass:
                raise ValueError(f"Unknown option '{option_key}'")
            self._strategy : ConstraintStrategy = strategyClass(self.start_time, self.prescibed_time, self.cfg,
                                                                self.FD_model)
            self.b_strategy = True
            
    def set_condition(self, conditions):
        self.conditions = conditions
        
    def set_action_limit(self, neg_lim:torch.tensor, pos_lim:torch.tensor):
        if self.b_strategy is False:
            self.limit_AC = [neg_lim.view(1, 1, self.action_dim),pos_lim.view(1, 1, self.action_dim)]
            return
        self._strategy.set_action_limit(neg_lim, pos_lim)
        self.limit_AC = [neg_lim.view(1, 1, self.action_dim),pos_lim.view(1, 1, self.action_dim)]
        
    def get_action_limit(self):
        return self.limit_AC
        
    def set_state_limit(self, obs_center:List[torch.tensor], obs_radius:List[torch.tensor]):
        self.obs_center = obs_center
        self.obs_radius = obs_radius
        
        if self.b_strategy is False:
            return

        self._strategy.set_state_limit(obs_center, obs_radius)
        
    def get_state_limit(self):
        return self.obs_center, self.obs_radius
        
        
    def ode_func(self, t:torch.tensor, x:torch.tensor):
        """_summary_

        Args:
            t (torch.tensor): scalar
            x (torch.tensor): [batch, horizon, act_dim+state_dim]

        Returns:
            torch.tensor: [batch, horizon, act_dim+state_dim]
        """
        # flow model
        dx_dt = self.model(x=x, t=t)
        
        # guidance
        if self.guide_fn is not None:
            dx_dt += self.guide_fn(x, t, dx_dt, self.model)
        
        # CBF/CLF
        if self.b_strategy is True:
            dx_dt = self._strategy.solve(x, t, dx_dt, self.ACSCDCFlag)
        
        # make sure condition is maintained 
        dx_dt = apply_conditioning_from_conditioned_x(
                dx_dt, torch.zeros_like(x), self.conditions, self.action_dim
            )
            
        return dx_dt

    def forward(self, x:torch.tensor, t_span:torch.tensor, *args, **kwargs):
        """
        Args:
            x [batch, horizon, act_dim+state_dim], 
            t_span [ode_time_step]
        """
        assert len(t_span) > 1, "t_span must have at least 2 elements"
        
        self.t_flow = t_span
        
        with torch.set_grad_enabled(False):
            # Approximate ODE solution with numerical ODE solver
            sol_tspan = odeint(
                self.ode_func,
                x,
                t_span,
                method=self.ode_method,
                atol=self.atol,
                rtol=self.rtol,
            ) # time_span, batch, hor, dim
            
            # pick the last timestep
            sol = sol_tspan[-1].detach()
        
        if self.b_strategy is True:
            self._strategy.reset_constant_calculated()
        
        return sol


class FlowPolicy(nn.Module):
    """
    This class is a wrapper around a flow model that generates actions from ONE step of 
    the observed state. 

    The generation is guided with the value model using different guidance methods.

    Normalization:
        Input observation and output action are denormalized; Models' input and output 
        are normalized.
    """
    def __init__(
        self, 
        flow_model, value_model, normalizer, action_dim, state_dim, horizon, 
        cfg: FlowMatchingEvaluationConfig,
        guide_model=None,
        FD_model=None,
    ):
        super().__init__()
        self.flow_model = flow_model
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon
        self.nMap = GetMapIndex(cfg.env)
        self.FD_model = FD_model

        self.cfg = cfg
        self.value_model = value_model # we need this to return value
        self.guide_model = guide_model
        
        self.prescribed_time = cfg.ode_t_span[1] - (cfg.ode_t_span[1]-cfg.ode_t_span[0])/(cfg.ode_t_steps-1) + 0.0001
        self.ode_t = torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=cfg.device
            )
        self.dt = self.ode_t[1] - self.ode_t[0]
        # CBF/CLF
        self.solver = CBFCLFODESolver(
            self.flow_model, 
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            hor=self.horizon,
            start_time= cfg.start_time,
            prescribe_time=self.prescribed_time,
            dt=self.dt,
            guide_fn=None, 
            FD_model=FD_model,
            ode_method=self.cfg.ode_solver,
            ACSCDCFlag=self.cfg.ACSCDCFlag,
            cfg=self.cfg,
            nMap=self.nMap
        )
        
        self.solver.set_strategy(self.cfg.constraint_strategy)
        self.act_neg_lim_nor = torch.tensor(self.normalizer.normalize(torch.tensor([cfg.act_neg_lim,cfg.act_neg_lim]), 'actions'),device=cfg.device) # [2]
        self.act_pos_lim_nor = torch.tensor(self.normalizer.normalize(torch.tensor([cfg.act_pos_lim,cfg.act_pos_lim]), 'actions'),device=cfg.device) # [2]
        
        obs_center_norm = []
        obs_radius_norm = []
        for i in range(len(cfg.obs_center)):
            obs_center_norm.append(torch.tensor(self.normalizer.normalize(np.array([cfg.obs_center[i][0], cfg.obs_center[i][1], 0.0, 0.0]), 'observations'),device=cfg.device)[0:2])
            obs_radius_norm.append(torch.tensor(self.normalizer.normalize_scaling(np.array([cfg.obs_radius[i][0], cfg.obs_radius[i][1], 0.0, 0.0]), 'observations'),device=cfg.device)[0:2])
        
        self.solver.set_action_limit(self.act_neg_lim_nor, self.act_pos_lim_nor)
        self.solver.set_state_limit(obs_center_norm, obs_radius_norm)
        
        self.obs_center_nor, self.obs_radius_nor = self.solver.get_state_limit()
        
    
    def compute_state_violate_normed(self, normed_obs:torch.tensor):
        """compute how much the state violate the constraint

        Args:
            normed_obs (torch.tensor): [B, hor, state_dim]

        Returns:
            torch.tensor: [B, hor], negative value means violation
        """
        B = normed_obs.shape[0]
        hor = normed_obs.shape[1]
        violate_all = torch.zeros(B, hor, 2, device=self.cfg.device) # B, hor, 2
        
        pos = normed_obs[:,:,:2] # B, hor, 2
        
        if self.nMap == 1: #Umaze
            violate_all[:,:,0] = (((pos - self.obs_center_nor[0])/self.obs_radius_nor[0])**4).sum(dim=-1) - 1
            violate_all[:,:,1] = 1 - (((pos - self.obs_center_nor[1])/self.obs_radius_nor[1])**4).sum(dim=-1)
        else:
            for i in range(2):
                violate_all[:,:,i] = (((pos - self.obs_center_nor[i])/self.obs_radius_nor[i])**2).sum(dim=-1) - 1
                
        violate_all = violate_all.clamp(max=0.0) # make sure only negative value, if success then 0
                
        violate_acc = violate_all.sum(dim=-1) # B, hor
        return violate_acc
    
    def compute_action_violate_normed(self, normed_act:torch.tensor):
        """compute how much the action violate the constraint

        Args:
            normed_act (torch.tensor): [B, hor, action_dim]

        Returns:
            torch.tensor: [B, hor], negative value means violation
        """
        B = normed_act.shape[0]
        hor = normed_act.shape[1]
        violate_all = torch.zeros(B, hor, 4, device=self.cfg.device) # B, hor, 2
        
        act = normed_act # B, hor, 2
        
        violate_all[:,:,:2] = -self.act_neg_lim_nor + act #B, Hor, 2
        violate_all[:,:,2:] = self.act_pos_lim_nor - act #B, Hor, 2
        
        violate_all = violate_all.clamp(max=0.0) # make sure only negative value, if success then 0
                
        violate_acc = violate_all.sum(dim=-1) # B, hor
        return violate_acc
    

    def set_record_file(self, final_policy_file_path:str):
        self.final_policy_file_path = final_policy_file_path
    
    def __call__(self, conditions, batch_size=1, verbose=True):
        # assert batch_size == 1, "batch_size must be 1 for now"
        if self.cfg.guidance_method in ['ss']:
            return self.ss_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ['reject_ss']:
            return self.reject_ss(conditions, batch_size)
        elif self.cfg.guidance_method in ['gradient']:
            return self.gradient_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ['mc']:
            return self.mc_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ['no']:
            pass
        elif self.cfg.guidance_method in ['guidance_matching']:
            return self.learned_guidance_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ['sim_mc']:
            return self.sim_mc_guidance_forward(conditions, batch_size)
        else:
            raise ValueError(f"Unsupported guidance method: {self.cfg.guidance_method}")

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=None, 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )        
    
        x = torch.randn(batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)

        x = solver(
            x, 
            t_span=self.ode_t,
        ) # (B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, 'actions')

        normed_observations = x[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        if self.cfg.guidance_method != 'no':
            values = self.value_model(normed_observations, normed_actions)
        else:
            values = None

        trajectories = Trajectories(actions, observations, values)
        
        # TODO: Add more "guidance" methods, including sample and selection-based MPC
        actions = actions[0, 0] # simply get the first action in the first sample in the batch
        
        return actions, trajectories


    ### Sample and Selection MPC ###

    def ss_forward(self, conditions, batch_size=1):
        assert batch_size == 1, "batch_size must be 1 for now"
        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        self.solver.set_condition(conditions)
        
        start_time = time.time()

        x = torch.randn(self.cfg.ss_batch * batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B_ss * B, T, C)
        conditions = to_torch(conditions, device=x.device) # {'0': tensor (B, C)}
        conditions = utils.apply_dict(lambda x: x.repeat(self.cfg.ss_batch, 1), conditions)
        x = apply_conditioning(x, conditions, self.action_dim)

        x = self.solver(
            x, 
            t_span=self.ode_t,
        ) # (B_ss * B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        normed_observations = x[:, :, self.action_dim:]
        
        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1)) # (B_ss * B, T, 1)
        values = values[:, -1, 0] # (B_ss * B)
        values = values.reshape(self.cfg.ss_batch, batch_size) # (B_ss, B)
        best_idx = values.argmax(dim=0).to(self.cfg.device) # (B,)
        
        # to construct trajectories, torch
        best_values = values[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B,)
        best_values = best_values[0] # (1,), select the first sample in the batch
        
        # from torch to np before pass to normalizer
        observations = self.normalizer.unnormalize(to_np(normed_observations), 'observations') #np
        best_observations = observations.reshape(self.cfg.ss_batch, batch_size, self.horizon, self.state_dim)[best_idx, np.arange(batch_size)] # (B, Hor, obs_dim)
        best_observations = best_observations[0] # (Hor, obs_dim), select the first sample in the batch
        
        actions = self.normalizer.unnormalize(to_np(normed_actions), 'actions') #np, [B,hor,act_dim]
        best_actions = actions.reshape(self.cfg.ss_batch, batch_size, self.horizon, self.action_dim)[best_idx, np.arange(batch_size)] # (1, T, C)

        trajectories = Trajectories(best_actions[0], best_observations, to_np(best_values))

        # output actions
        actions = best_actions[0, 0] # (act_dim), simply get the first action in the first sample in the batch
        
        end_time = time.time()
        comp_time = end_time - start_time
        Record_violation_into_file(x[best_idx:best_idx+1], self.action_dim, self.nMap,
                                self.obs_center_nor, self.obs_radius_nor, self.solver.get_action_limit(), 
                                self.FD_model, self.final_policy_file_path)

        return actions, trajectories, comp_time
    
    def reject_ss(self, conditions, batch_size=1):
        """find the most safe traj
        """
        assert batch_size == 1, "batch_size must be 1 for now"
        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        self.solver.set_condition(conditions)
        
        start_time = time.time()

        x = torch.randn(self.cfg.ss_batch * batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B_ss * B, T, C)
        conditions = to_torch(conditions, device=x.device) # {'0': tensor (B, C)}
        conditions = utils.apply_dict(lambda x: x.repeat(self.cfg.ss_batch, 1), conditions)
        x = apply_conditioning(x, conditions, self.action_dim)

        x = self.solver(
            x, 
            t_span=self.ode_t,
        ) # (B_ss * B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        normed_observations = x[:, :, self.action_dim:]
        
        # compute violation
        state_violate = self.compute_state_violate_normed(normed_observations) # (B_ss * B, H)
        action_violate = self.compute_action_violate_normed(normed_actions) # (B_ss * B, H)
        
        total_violate = (state_violate + action_violate).sum(-1) # (B_ss * B)
        
        best_violate_value, best_idx = torch.max(total_violate, dim=0) # more positive -> less violation
        
        # from torch to np before pass to normalizer
        observations = self.normalizer.unnormalize(to_np(normed_observations), 'observations') #np
        best_observations = observations.reshape(self.cfg.ss_batch, batch_size, self.horizon, self.state_dim)[best_idx, np.arange(batch_size)] # (B, Hor, obs_dim)
        best_observations = best_observations[0] # (Hor, obs_dim), select the first sample in the batch
        
        actions = self.normalizer.unnormalize(to_np(normed_actions), 'actions') #np, [B,hor,act_dim]
        best_actions = actions.reshape(self.cfg.ss_batch, batch_size, self.horizon, self.action_dim)[best_idx, np.arange(batch_size)] # (1, T, C)

        trajectories = Trajectories(best_actions[0], best_observations, to_np(best_violate_value))

        # output actions
        actions = best_actions[0, 0] # (act_dim), simply get the first action in the first sample in the batch
        
        end_time = time.time()
        comp_time = end_time - start_time
        Record_violation_into_file(x[best_idx:best_idx+1], self.action_dim, self.nMap,
                                self.obs_center_nor, self.obs_radius_nor, self.solver.get_action_limit(), 
                                self.FD_model, self.final_policy_file_path)

        return actions, trajectories, comp_time


    ### Taylor Expansion Approximate Gradient Guidance ###

    def get_gradient_guidance_model(self, value_model, schedule_fn, scale, grad_at='x_1', grad_to='x_1'):
        """
        Return the guidance model for the flow model.
        """
        assert self.cfg.guidance_method in ['gradient'], f"Unsupported guidance method: {self.cfg.guidance_method}"

        def guide_fn(x, t, dx_dt, flow_model):
            if grad_at == 'x_t':
                value = value_model(x)[:, -1, 0] # (B, T, 1) -> (B,)
            elif grad_at == 'x_1':
                x1_pred = x + (1 - t) * dx_dt
                value = value_model(x1_pred)[:, -1, 0] # (B, T, 1) -> (B,)
            else:
                raise ValueError(f"Unsupported gradient compute at: {grad_at}")
            if grad_to == 'x_t':
                grad = torch.autograd.grad([value.sum()], [x])[0]
            elif grad_to == 'x_1':
                assert grad_at == 'x_1', "cannot compute gradient wrt x_1 when grad_at is x_t"
                grad = torch.autograd.grad([value.sum()], [x1_pred])[0]
            else:
                raise ValueError(f"Unsupported gradient compute at: {grad_to}")
            return grad * scale * schedule_fn(t)
        return guide_fn

    def get_scheduler(self, schedule_fn):
        """
        Return the scheduler for the gradient guidance.
        """
        if schedule_fn == 'const':
            return lambda x: x
        elif schedule_fn == 'linear_decay':
            return lambda x: 1 - x
        elif schedule_fn == 'cosine_decay':
            return lambda x: 0.5 * (1 + torch.cos(x * math.pi))
        elif schedule_fn == 'exp_decay':
            return lambda x: (torch.exp(-x) - math.exp(-1)) / (1 - math.exp(-1))
        else:
            raise ValueError(f"Unsupported gradient schedule: {schedule_fn}")

    def gradient_forward(self, conditions, batch_size=1):
        """
        Use gradient guidance to generate actions.
        """
        # assert batch_size == 1, "batch_size must be 1 for now"
        assert self.cfg.guidance_method == 'gradient', f"guidance_method must be gradient, but got {self.cfg.guidance_method}"

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=self.get_gradient_guidance_model(
                self.value_model, 
                schedule_fn=self.get_scheduler(self.cfg.grad_schedule), 
                scale=self.cfg.grad_scale, 
                grad_at=self.cfg.grad_compute_at, 
                grad_to=self.cfg.grad_wrt
            ), 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )        
    
        x = torch.randn(batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)

        x = solver(x, t_span=torch.linspace(
            *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
        )) # (B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, 'actions')

        normed_observations = x[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1))
        trajectories = Trajectories(actions, observations, values)
        
        # TODO: Add more "guidance" methods, including sample and selection-based MPC
        actions = actions[0, 0] # simply get the first action in the first sample in the batch
        
        return actions, trajectories
    

    ### Monte-Carlo Approximate Guidance ###

    def _get_cached_ot_cfm_plan(self):
        if self.cached_ot_cfm_plan is None:
            raise ValueError("No cached OT-CFM plan found")
        return self.cached_ot_cfm_plan

    def _save_cached_ot_cfm_plan(self, x0_, x1_):
        self.cached_ot_cfm_plan = (x0_, x1_)

    def _remove_cached_ot_cfm_plan(self):
        self.cached_ot_cfm_plan = None

    def get_mc_guide_fn(self, x1, cached_v=None):
        """
        Compute the gradient guidance for the flow model.
        I think we only need to implement CFM and OT-CFM with Gaussian paths

        Args:
            x1_: Tensor, shape (B, T, C)
        """

        def cfm_log_p_t1(x1, xt, t, epsilon):
            # xt = t x1 + (1 - t) x0 -> x0 = xt / (1 - t) - t / (1 - t) x1
            x1 = x1.flatten(1) # (B, T * C)
            xt = xt.flatten(1) # (B, T * C)
            mu_t = t * x1 # (B, T * C)
            sigma_t = (1 - t + epsilon)
            log_p1t = torch.distributions.MultivariateNormal(
                mu_t, torch.eye(mu_t.shape[1], device=mu_t.device) * sigma_t
            ).log_prob(xt) # (B, T * C)
            return log_p1t
        
        def ot_cfm_log_p_tz(x0, x1, xt, t, std):
            """ 
            Args:
                std: float, g.t.: 0. Too small: requires large mc_batch_size; Too large: inaccurate
            """
            # xt = t x1 + (1 - t) x0 -> x0 = xt / (1 - t) - t / (1 - t) x1
            x0 = x0.flatten(1) # (B, T * C)
            x1 = x1.flatten(1) # (B, T * C)
            xt = xt.flatten(1) # (B, T * C)
            mean = t * x1 + (1 - t) * x0 # (B, T * C)
            log_p1t = torch.distributions.MultivariateNormal(
                mean, torch.eye(mean.shape[1], device=mean.device) * std
            ).log_prob(xt)
            return log_p1t

        def guide_fn(x, t, dx_dt, model):
            """
            Args:
                t: float
                x: Tensor, shape (b, T, C)
                dx_dt: Tensor, shape (b, T, C)
            """
            # estimate E (e^{-J} / Z - 1) * u
            MC_EP = self.cfg.mc_ep
            MC_B = self.cfg.mc_batch_size
            assert MC_B == x1.shape[0], "MC_B must be the same as the number of samples in x1"
            SCALE = self.cfg.mc_scale
            OT_STD = self.cfg.mc_ot_std
            b = x.shape[0]
            x_ = x.repeat(MC_B, 1, 1) # (MC_B * b, T, C)
            x1_ = x1.unsqueeze(0).repeat(b, 1, 1, 1).permute(1, 0, 2, 3).reshape(-1, *x1.shape[1:]) # (MC_B * b, T, C)
            
            if self.cfg.flow_matching_type == 'cfm':
                log_p_t1_x = cfm_log_p_t1(x1_, x_, t, epsilon=MC_EP) # (MC_B * b)
                
                if cached_v is None:
                    v_ = self.value_model(x1_)[:, -1, 0]
                else:
                    v_ = cached_v.clone()
                
                if self.cfg.mc_linear_J:
                    J_ = SCALE * v_ # value model output is (B, T, 1) but only the last step is used. J_: (MC_B * b)
                    if self.cfg.mc_self_normalize:
                        J_ = ((J_ - J_.mean()) / (J_.std() + 1e-8)).clamp(0)
                else:
                    # self normalize
                    if self.cfg.mc_self_normalize:
                        v_ = (v_ - v_.mean()) / (v_.std() + 1e-8)
                    J_ = torch.exp(SCALE * v_) # value model output is (B, T, 1) but only the last step is used. J_: (MC_B * b)
                
                log_p_t1_x = log_p_t1_x.reshape(MC_B, b, 1, 1)
                log_p_t_x = log_p_t1_x.logsumexp(0) - torch.log(torch.tensor(MC_B)) # (MC_B, B, 1, 1) -> (B, 1, 1)
                # Z = (p_t1_x * J_).reshape(MC_B, b, 1, 1).mean(0) / (p_t_x + 1e-8) # (MC_B, B, 1, 1) -> (B, 1, 1)
                log_p_t1_x_times_J_ = log_p_t1_x + torch.log(J_).reshape(MC_B, b, 1, 1) # (MC_B, b, 1, 1)
                log_Z = log_p_t1_x_times_J_.logsumexp(0) - torch.log(torch.tensor(MC_B)) - log_p_t_x # (B, 1, 1)
                Z = torch.exp(log_Z) # (B, 1, 1)

                u = (x1_ - x_) / (1 - t + MC_EP) # (MC_B * b, T, C)
                g = torch.exp(log_p_t1_x - log_p_t_x) \
                    * (J_.reshape(MC_B, b, 1, 1) / (Z + 1e-8) - 1) \
                    * u.reshape(MC_B, b, *x_.shape[1:]) # (MC_B, b, T, C)
                return g.mean(0) # (MC_B, B, T, C) -> (B, T, C)

            elif self.cfg.flow_matching_type == 'ot_cfm':
                try:
                    x0_, x1_ = self._get_cached_ot_cfm_plan()
                except:
                    x0_ = torch.randn(MC_B, *x.shape[1:], device=x.device) # (MC_B, T, C)
                    x0_, x1_ = OTPlanSampler(method='exact').sample_plan(x0_, x1_)
                    x0_ = x0_.unsqueeze(0).repeat(b, 1, 1, 1).permute(1, 0, 2, 3).reshape(-1, *x.shape[1:]) # (MC_B * b, T, C)
                    x1_ = x1_.unsqueeze(0).repeat(b, 1, 1, 1).permute(1, 0, 2, 3).reshape(-1, *x.shape[1:]) # (MC_B * b, T, C)
                    self._save_cached_ot_cfm_plan(x0_, x1_)
                log_p_t1_x = ot_cfm_log_p_tz(x0_, x1_, x_, t, std=OT_STD) # (MC_B * b)
                
                if cached_v is None:
                    v_ = self.value_model(x1_)[:, -1, 0]
                else:
                    v_ = cached_v.clone()
                
                if self.cfg.mc_linear_J:
                    J_ = SCALE * v_ # value model output is (B, T, 1) but only the last step is used. J_: (MC_B * b)
                    if self.cfg.mc_self_normalize:
                        J_ = ((J_ - J_.mean()) / (J_.std() + 1e-8)).clamp(0)
                else:
                    # self normalize
                    if self.cfg.mc_self_normalize:
                        v_ = (v_ - v_.mean()) / (v_.std() + 1e-8)
                    J_ = torch.exp(SCALE * v_) # value model output is (B, T, 1) but only the last step is used. J_: (MC_B * b)
                
                log_p_t1_x = log_p_t1_x.reshape(MC_B, b, 1, 1)
                log_p_t_x = log_p_t1_x.logsumexp(0) - torch.log(torch.tensor(MC_B)) # (MC_B, B) -> (B, 1, 1)
                # Z = (p_t1_x * J_).reshape(MC_B, b, 1, 1).mean(0) / (p_t_x + 1e-8) # (MC_B, B) -> (B, 1, 1)
                log_p_t1_x_times_J_ = log_p_t1_x + torch.log(J_).reshape(MC_B, b, 1, 1) # (MC_B, b, 1, 1)
                log_Z = log_p_t1_x_times_J_.logsumexp(0) - torch.log(torch.tensor(MC_B)) - log_p_t_x # (B, 1, 1)
                Z = torch.exp(log_Z) # (B, 1, 1)

                u = x1_ - x0_ # (MC_B * b, T, C)
                g = torch.exp(log_p_t1_x - log_p_t_x) \
                    * (J_.reshape(MC_B, b, 1, 1) / (Z + 1e-8) - 1) \
                    * u.reshape(MC_B, b, *x_.shape[1:]) # (MC_B, b, T, C)
                return g.mean(0) # (MC_B, B, T, C) -> (B, T, C)
            else:
                raise ValueError(f"Unsupported flow matching type: {self.cfg.flow_matching_type}")
        return guide_fn
    
    def mc_forward(self, conditions, batch_size=1):
        # assert batch_size == 1, "env batch_size must be 1 for now" # but SS_B can be > 1
        if batch_size > 1:
            print("WARNING: batch_size > 1 for MC, this is not tested")
        assert self.cfg.guidance_method == 'mc', f"guidance_method must be mc, but got {self.cfg.guidance_method}"
        
        b = self.cfg.mc_ss

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # first, sample support set x1 ~ p_1(x)
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=None, 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )
        x = torch.randn(self.cfg.mc_batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)
        with torch.no_grad():
            x1_support = solver(x, t_span=torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
            )) # (MC_B, T, C)
        
        # Then sample guided x1 ~ p_1(x) e^{R(x)} / Z 
        # precompute the value model output for the support set
        x1_support_rep = x1_support.unsqueeze(0).repeat(b * batch_size, 1, 1, 1).permute(1, 0, 2, 3).reshape(-1, *x1_support.shape[1:]) # (MC_B * b, T, C)
        v_support = self.value_model(x1_support_rep)[:, -1, 0].detach() # (MC_B * b)
        
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=self.get_mc_guide_fn(x1_support, cached_v=v_support), 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )
        x = torch.randn(b * batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)
        with torch.no_grad():
            x = solver(x, t_span=torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
            )) # (B, T, C)

        self._remove_cached_ot_cfm_plan()

        normed_actions = x[:, :, :self.action_dim]
        actions = torch.tensor(self.normalizer.unnormalize(normed_actions, 'actions'), device=self.cfg.device)

        normed_observations = x[:, :, self.action_dim:]
        observations = torch.tensor(self.normalizer.unnormalize(normed_observations, 'observations'), device=self.cfg.device) # NOTE: we do need to make torch tensor, otherwise the indexing later will be wrong
        
        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1)) # (B_ss * B, T, 1)
        values = values[:, -1, 0] # (B_ss * B)
        values = values.reshape(b, batch_size) # (B_ss, B)
        best_idx = values.argmax(dim=0).to(self.cfg.device) # (B,)
        
        # to construct trajectories
        best_values = values[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B,)
        best_observations = observations.reshape(b, batch_size, self.horizon, self.state_dim)[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B, T, C)
        best_actions = actions.reshape(b, batch_size, self.horizon, self.action_dim)[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B, T, C)
        
        trajectories = Trajectories(to_np(best_actions), to_np(best_observations), to_np(best_values))

        # output actions
        actions = actions.reshape(b, batch_size, self.horizon, self.action_dim)[best_idx, torch.arange(batch_size, device=self.cfg.device)] # (B, T, C)
        actions = to_np(actions[0, 0]) # (C,), simply get the first action in the first sample in the 
        
        return actions, trajectories


    ### Learned Guidance ###

    def get_learned_guidance_model(self, conditions):
        """
        Return the guidance model for the flow model.
        """
        def guide_fn(x, t, dx_dt, flow_model):
            if self.cfg.guide_matching_type != 'grad_z':
                with torch.no_grad():
                    guidance = self.guide_model(x, t) # input like flow model. x(B, T, C), t (,) or (B, )
            else:
                logz = self.guide_model(x, t)[:, -1, 0] # output is model_z output
                guidance = torch.autograd.grad([logz.sum()], [x])[0].detach()

            return guidance * self.cfg.guide_inference_scale
        return guide_fn

    def learned_guidance_forward(self, conditions, batch_size=1):
        """
        Use learned guidance to generate actions.
        """
        # assert batch_size == 1, "batch_size must be 1 for now"
        assert self.cfg.guidance_method == 'guidance_matching', f"guidance_method must be learned, but got {self.cfg.guidance_method}"
        assert self.guide_model is not None, "guide_model is not provided"

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=self.get_learned_guidance_model(conditions), 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )        
    
        x = torch.randn(batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)

        x = solver(x, t_span=torch.linspace(
            *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
        )) # (B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, 'actions')

        normed_observations = x[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1))
        trajectories = Trajectories(actions, observations, values)
        
        # TODO: Add more "guidance" methods, including sample and selection-based MPC
        actions = actions[0, 0] # simply get the first action in the first sample in the batch
        
        return actions, trajectories
    
    ### Simple p(z|x_1) MC guidance ###
    
    def get_sim_mc_guidance_model(self, value_model, schedule_fn, scale):
        """
        Return the guidance model for the flow model.
        """

        def guide_fn(x, t, dx_dt, flow_model):
            """
            Implements guidance following Eq. 12
            Args:
                t: flow time. float
                x: current sample x_t. Tensor, shape (b, dim)
                dx_dt: current predicted VF. Tensor, shape (b, dim)
                model: flow model. MLP
            """
            x1_pred = x + dx_dt * (1 - t) # (B, 2)

            x1 = torch.randn_like(
                x1_pred.unsqueeze(0).repeat(self.cfg.sim_mc_n, 1, 1, 1)
            ) * self.cfg.sim_mc_std + x1_pred # (cfg.sim_mc_n, B, C, T)
            values = value_model(x1.reshape(-1, *x1.shape[2:]))[:, -1, 0] # (cfg.sim_mc_n * B)
            if self.cfg.sim_mc_self_normalize:
                values = (values - values.mean()) / (values.std() + 1e-8) # (cfg.sim_mc_n * B)
            Jx1_ = torch.exp(
                self.cfg.sim_mc_J_scale * values
            ).reshape(self.cfg.sim_mc_n, -1) # (cfg.sim_mc_n, B)
            v = (x1 - x) / (1 - t + self.cfg.sim_mc_eps)  # Conditional VF v_{t|z} in Eq. 12 (cfg.sim_mc_n, B, C, T)
            Z = Jx1_.mean(0) + 1e-8  # Z in Eq. 12 (B,)
            g = (Jx1_ / Z - 1).reshape(self.cfg.sim_mc_n, -1, 1, 1) * v  # g in Eq. 12 (cfg.sim_mc_n, B, C, T)
            g = g.mean(0) # (B, C, T)
            return g * scale * schedule_fn(t)
        return guide_fn
    
    def sim_mc_guidance_forward(self, conditions, batch_size):
        """
        Use g^{\text{sim-MC}} guidance to generate actions.
        """
        # assert batch_size == 1, "batch_size must be 1 for now"

        # Only normalize the observation
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')

        # Generate actions
        solver = ConditionedODESolver(
            self.flow_model, 
            conditions, 
            guide_fn=self.get_sim_mc_guidance_model(
                self.value_model, 
                schedule_fn=self.get_scheduler(self.cfg.sim_mc_schedule), 
                scale=self.cfg.sim_mc_scale
            ), 
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )
    
        x = torch.randn(batch_size, self.horizon, self.action_dim + self.state_dim, device=self.cfg.device) # (B, T, C)
        x = apply_conditioning(x, to_torch(conditions, device=x.device), self.action_dim)

        x = solver(x, t_span=torch.linspace(
            *self.cfg.ode_t_span, self.cfg.ode_t_steps, device=x.device
        )) # (B, T, C)

        normed_actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, 'actions')

        normed_observations = x[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        values = self.value_model(torch.cat([normed_actions, normed_observations], dim=-1))
        trajectories = Trajectories(actions, observations, values)
        
        # TODO: Add more "guidance" methods, including sample and selection-based MPC
        actions = actions[0, 0] # simply get the first action in the first sample in the batch
        
        return actions, trajectories