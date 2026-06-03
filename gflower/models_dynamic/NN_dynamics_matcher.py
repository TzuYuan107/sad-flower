import torch
from torch import nn
from functorch import vmap, jacrev

class NNDynamicsJacRegMatcher:
    def __init__(
        self, 
        model: nn.Module
    ):
        self.model = model
        
    def loss(self, cur_obs, cur_act, next_obs, weight_jac_regular=0.1, compute_jacobian=True):
        """
        Args:
            x: (B, T, C), normalized
            cond: [ (time, state), ... ]; length T, state (B, C_state)
        """
        B, H, s_dim = cur_obs.shape
        _, _, a_dim = cur_act.shape
        
        # Flatten batch and horizon: [B*H, s_dim] and [B*H, a_dim]
        cur_obs_flat = cur_obs.reshape(-1, s_dim)
        cur_act_flat = cur_act.reshape(-1, a_dim)
        
        next_obs_pred = self.model(cur_obs, cur_act)
        forward_loss = torch.nn.functional.mse_loss(next_obs_pred, next_obs).mean()
        jac_reg_loss = 0
        
        # train with jacobian regularization
        if compute_jacobian:
            def wrapped(s_, a_): return self.model(s_.unsqueeze(0), a_.unsqueeze(0)).squeeze(0)
            jac_s_flat = vmap(jacrev(wrapped, argnums=0))(cur_obs_flat, cur_act_flat)
            jac_a_flat = vmap(jacrev(wrapped, argnums=1))(cur_obs_flat, cur_act_flat)
            jac_reg_loss = ((jac_s_flat ** 2).mean() + (jac_a_flat ** 2).mean()) * weight_jac_regular
        
        All_loss = forward_loss + jac_reg_loss

        infos = {'All_loss': All_loss.mean().item(), 'Forward_loss': forward_loss.mean().item(), 'Jac_loss': jac_reg_loss.item() if compute_jacobian else 0.0}
        
        return All_loss, infos
    
class NNDynamicsMatcher:
    def __init__(
        self, 
        model: nn.Module
    ):
        self.model = model
        
    def loss(self, cur_obs, cur_act, next_obs):
        """
        Args:
            x: (B, T, C), normalized
            cond: [ (time, state), ... ]; length T, state (B, C_state)
        """
        next_obs_pred = self.model(cur_obs, cur_act)
        forward_loss = torch.nn.functional.mse_loss(next_obs_pred, next_obs).mean()
        
        infos = {'loss': forward_loss.mean().item()}
        
        # All_loss = forward_loss
        # infos = {'All_loss': All_loss.mean().item(), 'Forward_loss': All_loss.mean().item(), 'Jac_loss': All_loss.mean().item()}
        return forward_loss, infos