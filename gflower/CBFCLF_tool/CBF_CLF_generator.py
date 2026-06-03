"""
=================================================================================
CBF-CLF Quadratic Programming (QP) Constraint Framework
=================================================================================

This module implements a Control Barrier Function (CBF) and Control Lyapunov 
Function (CLF) based optimization framework for trajectory planning. The system 
solves a batch QP problem to generate control inputs that satisfy safety (CBF), 
convergence (CLF), and action constraints while minimizing deviation from a 
reference flow.

KEY CONCEPTS:
=============

1. QP OPTIMIZATION VARIABLE STRUCTURE
------------------------------------
The QP optimization variable is structured as a flattened trajectory:
    
    x_opt = [a₀ | a₁ s₁ | a₂ s₂ | ... | aₜ₋₁ sₜ₋₁]
    
where:
    - a_i ∈ ℝ^{action_dim}  (action at timestep i)
    - s_i ∈ ℝ^{state_dim}   (state at timestep i)
    - s₀ (initial state) is FIXED and not optimized
    
DIMENSION CALCULATION:
    - Total optimization variables: horizon * (action_dim + state_dim) - state_dim
    - Because we exclude s₀ (it's fixed from the problem input)
    
COLUMN LAYOUT (for horizon=4, action_dim=2, state_dim=4):
    [a₀ | a₁  s₁ | a₂  s₂ | a₃  s₃]
    [0-2| 2-4 4-8|8-10 10-14|14-16 16-20]
    
    - a₀: [0, 2)
    - Loop iteration i=0: a₁ [2, 4), s₁ [4, 8)
    - Loop iteration i=1: a₂ [8, 10), s₂ [10, 14)
    - Loop iteration i=2: a₃ [14, 16), s₃ [16, 20)

2. CBF CONSTRAINT STRUCTURE (State + Action Constraints)
--------------------------------------------------------
CBF constraints ensure safety: h(x) ≥ 0 by ḣ(x) + α(h(x)) ≥ 0 where h is the barrier function.

STRUCTURE:
    - ACTION CONSTRAINTS (AC): Applied only to FIRST timestep (a₀)
    - STATE CONSTRAINTS (SC): Applied to all timesteps (s₁, s₂, ..., sₜ₋₁)
    
ROW LAYOUT (for horizon=4, n_dim_ac=2, n_dim_sc=1, n_dim_scac=3):
    Row 0-1:    AC for a₀
    Row 2-4:    AC for a₁, SC for s₁
    Row 5-7:    AC for a₂, SC for s₂
    Row 8-10:   AC for a₃, SC for s₃
    
ASSIGNMENT STRATEGY:
    1. Initial: Assign AC for a₀ only
       G_flat[:, 0:n_dim_ac, 0:action_dim] = G_AC  # a₀ columns
    
    2. Loop (i=0..horizon-2): Assign AC + SC for each subsequent timestep
       For loop iteration i, assign:
       - AC for a_{i+1}:
         Rows: [n_dim_ac + i*n_dim_scac : n_dim_ac + i*n_dim_scac + n_dim_ac)
         Cols: [all_dim*i + action_dim : all_dim*i + 2*action_dim)
       - SC for s_{i+1}:
         Rows: [n_dim_ac + i*n_dim_scac + n_dim_ac : n_dim_ac + (i+1)*n_dim_scac)
         Cols: [all_dim*i + 2*action_dim : all_dim*i + 2*action_dim + nDimOpt_SC)
    
CRITICAL: SC constraints only apply to OPTIMIZABLE state dimensions (nDimOpt_SC),
          NOT the full state_dim. E.g., if only position is constrained but 
          velocity is not: nDimOpt_SC = 2 (x,y) while state_dim = 4 (x,y,vx,vy)

3. CLF CONSTRAINT STRUCTURE (Dynamics Constraint)
-------------------------------------------------
CLF constraints ensure convergence: V̇(x) ≤ -γV where V = 0.5*||s_{i+1} - f(s_i,a_i)||²

STRUCTURE:
    - SINGLE ROW for ALL dynamics constraints (accumulated)
    - MULTIPLE COLUMNS for each timestep's dynamics relationship
    
ROW LAYOUT:
    All CLF constraints go to a SINGLE row (either row 0 for OnlyDC mode,
    or row = hor*nDim_SCAC - nDim_state for ACSCDC mode)

COLUMN LAYOUT - Assignment Pattern:
    1. Initial: Assign coefficients for s₁ = f(s₀, a₀)
       Cols: [0 : action_dim) for a₀
       Cols: [2*action_dim : 2*action_dim + state_dim) for s₁ (next_s)
    
    2. Loop (i=0..horizon-3): Accumulate coefficients for subsequent dynamics
       For loop iteration i, add coefficients for s_{i+2} = f(s_{i+1}, a_{i+1}):
       Cols: [action_dim + i*all_dim : action_dim + i*all_dim + action_dim) for a_{i+1}
       Cols: [2*action_dim + i*all_dim : 2*action_dim + i*all_dim + state_dim) for s_{i+1}
       Cols: [3*action_dim + state_dim + i*all_dim : ...] for s_{i+2} (next_s)
    
EXAMPLE (horizon=4):
    G[DC, :] contains accumulated coefficients for:
    - a₀, s₁:      cols [0,2), [4,8)     (s₁ = f(s₀, a₀))
    - a₁, s₁, s₂:  cols [2,4), [4,8), [10,14)  (s₂ = f(s₁, a₁))
    - a₂, s₂, s₃:  cols [8,10), [10,14), [16,20)  (s₃ = f(s₂, a₂))
    - ... (accumulated into the SAME row)
    
CRITICAL DIFFERENCE FROM CBF:
    - CBF: Multiple rows, one per constraint
    - CLF: Single accumulated row, multiple time-indexed dynamics

=================================================================================
"""

from abc import ABC, abstractmethod
import torch
from torch.autograd import Variable
from qpth.qp import QPFunction, QPSolvers
from functorch import vmap, jacrev
from gflower.config.flow_matching import FlowMatchingEvaluationConfig
from typing import Dict, List, Tuple
from gflower.utils.debug import GetMapIndex, ConstructExpForwardModelNormed
import warnings
warnings.filterwarnings(
    'ignore',
    message="WARNING batched routines are designed for small sizes. It might be better to use the Native/Hybrid classical routines if you want good performance."
)

class CBFCLFFunctionGenerator(ABC):
    def __init__(self, start_time, prescibed_time, nDim_state, nDim_action, cfg:FlowMatchingEvaluationConfig,
                 FD_model=None):
        super().__init__()
        self.nDim_state = nDim_state
        self.nDim_action = nDim_action
        self.nDim_all = self.nDim_state + self.nDim_action
        self.c_CBF = 0.1
        self.c_PTCBF = 0.1
        self.c_PTCLF = cfg.CLF_c
        self.dummy_constraint = Variable(torch.Tensor())
        self.device = 'cpu'
        self.b_cPT_calculated = False
        self.start_time = start_time
        self.prescibed_time = prescibed_time
        self.FD_model = FD_model
        self.PT_ord1 = cfg.PT_ord1
        self.cfg = cfg
        self.PT_CBF_min = cfg.PT_CBF_min
        self.PT_CLF_min = cfg.PT_CLF_min
        self.d_robust = cfg.d_robust
        self.s_robust = cfg.s_robust
        self.a_robust = cfg.a_robust
        self.nMap = GetMapIndex(cfg.env)
        self.__calexplicitJac(cfg)
        self.b_expJ = False
        if (cfg.ACSCDCFlag == 1):
            self.b_expJ = True
        else:
            self.b_expJ = False
        
    def __calexplicitJac(self, cfg:FlowMatchingEvaluationConfig):
        self.expJac_s, self.expJac_a, self.exp_const = ConstructExpForwardModelNormed(cfg, cfg.ss_batch * cfg.batch_size, cfg.horizon-1)
        # [B, H-1, obs_dim, obs_dim], [B, H-1, obs_dim, act_dim]
    
    def expForwardModel_normed(self, state:torch.tensor, action: torch.tensor) -> torch.tensor:
        """_summary_

        Args:
            state (torch.tensor): [B, H, s_dim]
            action (torch.tensor): [B, H, a_dim]

        Returns:
            [s_dim]
        """
        Ax = self.expJac_s @ state.unsqueeze(-1) # [B, H, obs_dim, 1]
        Bu = self.expJac_a @ action.unsqueeze(-1) # [B, H, obs_dim, 1]
        return Ax.squeeze(-1) + Bu.squeeze(-1) + self.exp_const
        
        
    def set_device(self, device:str):
        self.device = device
        
    def ResetConstantCalculated(self):
        self.b_cPT_calculated = False
        
    def _FD(self, x:torch.tensor) -> torch.tensor:
        """_summary_

        Args:
            x (torch.tensor): [s_dim+a_dim]

        Returns:
            [s_dim]
        """
        a, s = x[:self.nDim_action], x[self.nDim_action:]
        return self.FD_model(s, a)
    
    
    
    def _get_jacobian(self, x_flat:torch.tensor, state_dim:int, action_dim:int, batch:int, hor:int):
        """_summary_

        Args:
            x_flat (torch.tensor): [batch*(hor), state_dim+action_dim]
            state_dim (int): _description_
            batch (int): _description_
            hor (int): _description_

        Returns:
            _type_: _description_
        """
        jac_all = vmap(jacrev(self._FD))(x_flat)

        # Step 5: Split into ∂f/∂s and ∂f/∂a
        jac_a = jac_all[:, :, :action_dim]   # shape: [B*H, state_dim, act_dim]
        jac_s = jac_all[:, :, action_dim:]   # shape: [B*H, state_dim, state_dim]

        # Step 6: Reshape back to [batch, hor, ...]
        jac_a = jac_a.view(batch, hor, state_dim, action_dim) # shape: [B, H, state_dim, act_dim]
        jac_s = jac_s.view(batch, hor, state_dim, state_dim) # shape: [B, H, state_dim, state_dim]
        
        return jac_s, jac_a
    
    def _get_explicit_jacobian(self):
        """_summary_

        Returns:
            jac_s (torch.tensor): [B, H, state_dim, act_dim]
            jac_a (torch.tensor): [B, H, state_dim, state_dim]
        """
        return self.expJac_s, self.expJac_a
    
    def _PTCBF_TimeGain_CBF(self,t:torch.tensor) -> float:
        phi_t_dom = max(torch.square(self.prescibed_time-t),self.PT_CBF_min)
        phi_t = (1/phi_t_dom)+self.PT_ord1*t
        return phi_t
    
    def _PTCBF_TimeGain_CLF(self,t:torch.tensor) -> float:
        phi_t_dom = max(torch.square(self.prescibed_time-t),self.PT_CLF_min)
        phi_t = (1/phi_t_dom)+self.PT_ord1*t
        return phi_t
    
    def _flattern_QP_batch(self,Q:torch.tensor,P:torch.tensor,G:torch.tensor,H:torch.tensor) -> torch.tensor:
        """flattern QPFunction component

        Args:
            Q (torch.tensor): [batch, hor, opt_dim, opt_dim]
            P (torch.tensor): [batch, hor, opt_dim]
            G (torch.tensor): [batch, hor, constraint_dim, opt_dim]
            H (torch.tensor): [batch, hor, constraint_dim]
            A (torch.tensor): [batch, hor, 0, opt_dim]
            B (torch.tensor): [batch, hor, 0]

        Returns:
            Q_f (torch.tensor): [batch*hor, opt_dim, opt_dim]
            P_f (torch.tensor): [batch*hor, opt_dim]
            G_f (torch.tensor): [batch*hor, constraint_dim, opt_dim]
            H_f (torch.tensor): [batch*hor, constraint_dim]
            A_f (torch.tensor): [batch*hor, 0, opt_dim]
            B_f (torch.tensor): [batch*hor, 0]
            
        """
        batch, hor, constraint_dim, opt_dim = G.shape
        nflattern_size = batch * hor
        Q_f = Q.reshape(nflattern_size, opt_dim, opt_dim)
        P_f = P.reshape(nflattern_size, opt_dim)
        G_f = G.reshape(nflattern_size, constraint_dim, opt_dim)
        H_f = H.reshape(nflattern_size, constraint_dim)
        
        return Q_f, P_f, G_f, H_f
    
    def _Get_CFSolution_CBF(self,G_flat:torch.tensor, P_flat:torch.tensor, H_flat:torch.tensor, 
                               batch:int, hor:int, 
                               dx_dt:torch.tensor,
                               nC_opt_dim:int, nidx_C_in_all:int, nidx_C_in_C:int) -> torch.tensor:
        """get solution for CBF optimization problem by closed loop

        Args:
            G_flat (torch.tensor): [batch*hor, constraint_dim, opt_dim]
            P_flat (torch.tensor): [batch*hor, constraint_dim]
            H_flat (torch.tensor): [batch*hor, opt_dim]
            batch (int): 
            hor (int): 
            dx_dt (torch.tensor): [batch, horizon, act_dim+state_dim]
            nC_opt_dim (int): dim of optimization variable
            nidx_C_in_all (int): the index of constraint corresponding to all dimension (state+act)
            nidx_C_in_C (int): the index of constraint corresponding to Constraint (state:1, act:3)

        Returns:
            torch.tensor: [batch, horizon, act_dim+state_dim]
        """
        total_batch, c_dim, nC_opt_dim = G_flat.shape
        ################ mask + inverse
        v=-P_flat.unsqueeze(-1) #[batch*hor,opt_dim,1]
        Gv = torch.bmm(G_flat, v) #[batch*hor,C_dim,1]
        
        # Step 1: Compute residuals r = G @ v_theta - h  -> [batch, 7, 1]
        r = Gv - H_flat.unsqueeze(-1) #[batch*hor,C_dim,1]

        # Step 2: Always keep index 0
        idx0 = torch.zeros(total_batch, 1, dtype=torch.long, device=self.device)  # shape [batch*hor, 1]

        # Step 3: Compare index pairs
        r_flat = r.squeeze(-1)  # [batch*hor, 7]
        pairs = [(1, 4), (2, 5), (3, 6)]

        selected_idxs = [] # [batch*hor, 3]
        for i, j in pairs:
            more_violated = torch.where(r_flat[:, i] > r_flat[:, j], i, j)
            selected_idxs.append(more_violated.unsqueeze(1))  # [batch*hor, 1]

        # Step 4: Concatenate selected indices with index 0 -> shape [batch*hor, 4]
        mask_indices = torch.cat([idx0] + selected_idxs, dim=1)  # [batch*hor, 4]

        # Step 5: Gather G and h by advanced indexing
        batch_indices = torch.arange(total_batch).unsqueeze(1).expand(-1, 4)  # [batch*hor, 4]
        G_masked = G_flat[batch_indices, mask_indices]  # [batch*hor, 4, 4]
        H_masked = H_flat.unsqueeze(-1)[batch_indices, mask_indices]  # [batch*hor, 4, 1]`
        
        Gv_mask = torch.bmm(G_masked, v) #[batch*hor,mask_dim,1]
        G_T_MASK = G_masked.transpose(1, 2) #[batch*hor,opt_dim,mask_dim]
        GGT_MASK = torch.bmm(G_masked, G_T_MASK)

        landa = torch.linalg.solve(GGT_MASK, Gv_mask-H_masked)
        v_correction = -torch.bmm(G_T_MASK,torch.clamp(landa, min=0)).view(batch, hor, nC_opt_dim) #[batch,hor,SC_opt_dim]
        
        sol_C = dx_dt
        sol_C[:,:,nidx_C_in_all] = sol_C[:,:,nidx_C_in_all] + v_correction[:,:,nidx_C_in_C]
        
        return sol_C
        
    
    @abstractmethod
    def set_action_limit(self, neg_lim, pos_lim):
        pass
    
    def set_state_limit(self, obs_center:List[torch.tensor], obs_radius:List[torch.tensor]):
        pass
    
    def GetQP_PT_ACSC(self, x:torch.tensor, t:torch.tensor, dx_dt:torch.tensor):
        """Get Prescibed-time QP optimization solution for state constraint and action constraint

        Args:
            x (torch.tensor): [batch, horizon, act_dim+state_dim]
            t (torch.tensor): scalar
            dx_dt (torch.tensor): [batch, horizon, act_dim+state_dim]
        """
        pass
    
    
    def GetQP_PT_ACSCDC(self, x:torch.tensor, t:torch.tensor, dx_dt:torch.tensor, ACSCDCFlag:int=1):
        """Get Prescibed-time QP optimization solution for state, action constraint, and dynamic constraint

        Args:
            x (torch.tensor): [batch, horizon, act_dim+state_dim]
            t (torch.tensor): scalar
            dx_dt (torch.tensor): [batch, horizon, act_dim+state_dim]
        """
        pass
        
        

class OpenMazeSystem(CBFCLFFunctionGenerator):
    def __init__(self, start_time, prescibed_time, cfg:FlowMatchingEvaluationConfig, FD_model=None):
        super().__init__(start_time, prescibed_time, 4, 2, cfg, FD_model=FD_model)
        
        ######################### State Constraint (SC)
        self.nOptIdx_SC_in_all = [2,3] # SC in all variable
        self.nDimOpt_SC = len(self.nOptIdx_SC_in_all) # 2
        self.nDim_SC = len(cfg.obs_center) # 1
        
        ######################### Action Constraint (AC)
        self.nOptIdx_AC_in_all = [0,1] # AC in all variable
        self.nDimOpt_AC = len(self.nOptIdx_AC_in_all) #2
        self.limit_AC = [-1*torch.ones(self.nDimOpt_AC),1*torch.ones(self.nDimOpt_AC)]
        self.nDim_AC = sum(row.numel() for row in self.limit_AC) #4
        self.scale_AC_opt = [1,-1]
        
        ######################### SC + AC
        self.nDimOpt_SCAC = self.nDimOpt_SC + self.nDimOpt_AC #4
        self.nDim_SCAC = self.nDim_SC + self.nDim_AC #5
        
        ######################### define opt index for SC, SCAC
        # SC_opt [x,y]
        self.nIdx_SC_in_SC = [0,1]
        # SCAC_opt [fx, fy, x, y] 
        self.nOptIdx_SC_in_SCAC = [2,3] 
        self.nOptIdx_AC_in_SCAC = [0,1] 
        self.nOptIdx_SCAC_in_SCAC = self.nOptIdx_AC_in_SCAC + self.nOptIdx_SC_in_SCAC
        
        ######################### define index in all variable e.g. dx_dt
        self.nOptIdx_SCAC_in_all = self.nOptIdx_AC_in_all + self.nOptIdx_SC_in_all
        
        ######################### Dynamic Constraint (DC) without horizon
        self.nDim_DC = 1
        
        # SC + AC + DC
        self.nDim_ACSCDC = self.nDim_SC + self.nDim_AC + self.nDim_DC
        self.nDim_ACDC = self.nDim_AC + self.nDim_DC
        
        self.nDimOpt_ACSCDC = self.nDim_all
        self.G_AC = -torch.tensor([[1,0],[0,1],[-1,0],[0,-1]], device=self.device)
    
    def set_action_limit(self, neg_lim:torch.tensor, pos_lim:torch.tensor):
        """_summary_

        Args:
            neg_lim (torch.tensor): [action_dim]
            pos_lim (torch.tensor): [action_dim]
        """
        self.limit_AC = [neg_lim,pos_lim]
        
    def set_state_limit(self, obs_center:List[torch.tensor], obs_radius:List[torch.tensor]):
        """_summary_

        Args:
            obs_center list[(torch.tensor)]: [(obs1_x, obs1_y), (obs2_x, obs2_y),...]
            obs_radius list[((torch.tensor)]: [(obs1_a, obs1_b), (obs2_a, obs2_b),...]
        """
        self.obs_center = obs_center
        self.obs_radius = obs_radius
        
    def __computeSC_H(self, x_opt:torch.tensor):
        """generate h, for state constraint

        Args:
            x_opt (torch.tensor): [batch, hor, n_SC_opt_dim]

        Returns:
            h_x (torch.tensor): [batch, hor, state_constraint_dim]
        """
        B, H, _ = x_opt.shape
        h_x = torch.zeros(B, H, self.nDim_SC,device=self.device)
        if self.nMap == 1: #Umaze
            h_x[:,:,0] = (((x_opt - self.obs_center[0])/self.obs_radius[0])**4).sum(dim=-1) - 1
            h_x[:,:,1] = 1 - (((x_opt - self.obs_center[1])/self.obs_radius[1])**4).sum(dim=-1)
        else:
            for i in range(self.nDim_SC):
                h_x[:,:,i] = (((x_opt - self.obs_center[i])/self.obs_radius[i])**2).sum(dim=-1) - 1
        return h_x # batch, horizon, n_SC_dim

    def __computeSC_HGrad(self, x_opt:torch.tensor):
        """generate h_grad for state constraint

        Args:
            x_opt (torch.tensor): [batch, hor, n_SC_opt_dim]

        Returns:
            h_grad_x (torch.tensor): [batch, hor, state_constraint_dim, opt_dim]
        """
        B, H, _ = x_opt.shape
        h_grad_x = torch.zeros(B,H,self.nDim_SC, self.nDimOpt_SC,device=self.device)
        
        if self.nMap == 1: #Umaze
            h_grad_x[:,:,0,:] = 4*((x_opt-self.obs_center[0])/((self.obs_radius[0])))**3/self.obs_radius[0]
            h_grad_x[:,:,1,:] = -4*((x_opt-self.obs_center[1])/((self.obs_radius[1])))**3/self.obs_radius[1]
        else:
            for i in range(self.nDim_SC):
                h_grad_x[:,:,i,:] = 2*(x_opt-self.obs_center[i])/((self.obs_radius[i])**2)

        return h_grad_x
    
    def __computeSC_H_HGrad(self, x:torch.tensor) -> torch.tensor:
        """generate h, h_grad for state constraint

        Args:
            x (torch.tensor): [batch, hor, act_dim + state_dim]

        Returns:
            h_x (torch.tensor): [batch, hor, state_constraint_dim]
            h_grad_x (torch.tensor): [batch, hor, state_constraint_dim, opt_dim]
        """
        x_opt = x[:,:,self.nOptIdx_SC_in_all] # batch, hor, n_SC_opt_dim
        return self.__computeSC_H(x_opt), self.__computeSC_HGrad(x_opt)
    
    def __computeAC_H(self, x_opt:torch.tensor) -> torch.tensor:
        """generate h for action constraint

        Args:
            x (torch.tensor): [batch, hor, n_AC_opt_dim]

        Returns:
            h_x (torch.tensor): [batch, hor, action_constraint_dim]
        """
        h_x_lb = -self.limit_AC[0] + self.scale_AC_opt[0] * x_opt #batch, horizon, n_AC_dim/2, (n_AC_dim/2 = n_AC_opt_dim in this case) 
        h_x_ub = self.limit_AC[1] + self.scale_AC_opt[1] * x_opt #batch, horizon, n_AC_dim/2
        return torch.cat( [h_x_lb, h_x_ub], dim=2 ) # batch, horizon, n_AC_dim
    
    def __computeAC_HGrad(self, x_opt:torch.tensor):
        """generate h_grad for action constraint

        Args:
            x (torch.tensor): [batch, hor, n_AC_opt_dim]

        Returns:
            h_grad_x (torch.tensor): [batch, hor, action_constraint_dim, opt_dim]
        """
        # default n_AC_dim/2 = n_AC_opt_dim because -a_i,bar<a_i<a_i,bar, this can be extended to more general constraint
        batch, hor, _ = x_opt.shape
        I = torch.eye(self.nDimOpt_AC, device=self.device).unsqueeze(0).unsqueeze(0).expand(batch, hor, self.nDimOpt_AC, self.nDimOpt_AC) 
        h_grad_lb = self.scale_AC_opt[0] * I # batch, hor, n_AC_dim/2, n_AC_opt_dim
        h_grad_ub = self.scale_AC_opt[1] * I # batch, hor, n_AC_dim/2, n_AC_opt_dim
        return torch.cat([h_grad_lb, h_grad_ub], dim=2) # batch, hor, n_AC_dim, n_AC_opt_dim
    
    def __computeAC_H_HGrad(self, x:torch.tensor):
        """generate h, h_grad for action constraint

        Args:
            x (torch.tensor): [batch, hor, act_dim + state_dim]

        Returns:
            h_x (torch.tensor): [batch, hor, action_constraint_dim]
            h_grad_x (torch.tensor): [batch, hor, action_constraint_dim, opt_dim]
        """
        x_opt = x[:,:,self.nOptIdx_AC_in_all] # batch, hor, n_AC_opt_dim
        return self.__computeAC_H(x_opt), self.__computeAC_HGrad(x_opt)
    
    def __computeSCAC_H_HGrad(self, x:torch.tensor):
        """generate h, h_grad for state constraint + action constraint

        Args:
            x (torch.tensor): [batch, hor, act_dim + state_dim]

        Returns:
            h_x (torch.tensor): [batch, hor, state_constraint_dim + action_constraint_dim]
            h_grad_x (torch.tensor): [batch, hor, state_constraint_dim + action_constraint_dim, opt_dim_state + opt_dim_action]
        """
        batch, hor, dim = x.shape
        
        h_x_AC, h_grad_x_AC = self.__computeAC_H_HGrad(x)
        h_x_SC, h_grad_x_SC = self.__computeSC_H_HGrad(x)
        h_x = torch.cat([h_x_AC, h_x_SC],dim=2) #batch, horizon, n_SC_dim + n_AC_dim
        
        h_grad_x = torch.zeros(batch, hor, self.nDim_SCAC, self.nDimOpt_SCAC).to(self.device) #batch, horizon, n_SC_dim + n_AC_dim, n_SC_opt_dim + n_AC_opt_dim
        h_grad_x[:,:,:self.nDim_AC,:self.nDimOpt_AC] = h_grad_x_AC
        h_grad_x[:,:,self.nDim_AC:, self.nDimOpt_AC:] = h_grad_x_SC
        
        
        return h_x, h_grad_x
    
    def __constructPTQPFunction(self, h_x:torch.tensor, h_grad_x:torch.tensor, dx_dt:torch.tensor, t:torch.tensor, nidx_optidx_in_all:list):
        """generate QPFunction component

        Args:
            h_x (torch.tensor): [batch, hor, constraint_dim]
            h_grad_x (torch.tensor): [batch, hor, constraint_dim, opt_dim]
            dx_dt (torch.tensor): [batch, hor, all_dim]
            t (torch.tensor): [scalar]
            nidx_optidx_in_all (list): opt_dim (idx for opt variable)

        process:    
            Q_QP (torch.tensor): [batch, hor, opt_dim, opt_dim]
            P_QP (torch.tensor): [batch, hor, opt_dim]
            G_QP (torch.tensor): [batch, hor, constraint_dim, opt_dim]
            H_QP (torch.tensor): [batch, hor, constraint_dim]
            A_QP (torch.tensor): [batch, hor, 0, opt_dim]
            B_QP (torch.tensor): [batch, hor, 0]

        Returns:
            Q_f (torch.tensor): [batch*hor, opt_dim, opt_dim]
            P_f (torch.tensor): [batch*hor, opt_dim]
            G_f (torch.tensor): [batch*hor, constraint_dim, opt_dim]
            H_f (torch.tensor): [batch*hor, constraint_dim]
            A_f (torch.tensor): [batch*hor, 0, opt_dim]
            B_f (torch.tensor): [batch*hor, 0]
        """
        batch, hor, dim_C, dim_opt = h_grad_x.shape
        
        # cost function
        I = torch.eye(dim_opt).to(self.device).unsqueeze(0).unsqueeze(0) #1, 1, opt_dim, opt_dim
        Q_QP = I.expand(batch, hor, dim_opt, dim_opt) # batch, hor, opt_dim, opt_dim
        P_QP = -dx_dt[:,:,nidx_optidx_in_all] # batch, hor, opt_dim
        
        # inequality
        if self.b_cPT_calculated is False:
            # no solution of QP
            # self.c_PTCBF = torch.clamp(-1/h_x * torch.einsum('b h c o, b h o -> b h c',
            #                  h_grad_x,
            #                  dx_dt[:,:,nidx_optidx_in_all]), min=0.1)
            self.c_PTCBF = torch.where(h_x >= 0,
                self.cfg.CBF_c[0],     # value where h_x > 0
                self.cfg.CBF_c[1]) 
            self.b_cPT_calculated = True
        G_QP = -h_grad_x # batch, hor, n_SC_dim, opt_dim
        H_QP = self.c_PTCBF * self._PTCBF_TimeGain_CBF(t) * (h_x) - self.s_robust # batch, hor, n_SC_dim
        
        # No equalities
        A_flat = self.dummy_constraint.to(self.device)
        B_flat = self.dummy_constraint.to(self.device)
        
        Q_flat, P_flat, G_flat, H_flat = self._flattern_QP_batch(Q_QP, P_QP, G_QP, H_QP)
        return Q_flat, P_flat, G_flat, H_flat, A_flat, B_flat

    def __constructPT_SCAC_coeff(self,x:torch.tensor, t:torch.tensor, dx_dt:torch.tensor):
        """construct the coefficient for QP/Closed Form for SCAC

        Args:
            x (torch.tensor): [batch, horizon, act_dim+state_dim]
            t (torch.tensor): scalar
            dx_dt (torch.tensor): [batch, horizon, act_dim+state_dim]
        """     
        h_x, h_grad_x = self.__computeSCAC_H_HGrad(x)
        Q_flat, P_flat, G_flat, H_flat, A_flat, B_flat = self.__constructPTQPFunction(h_x, h_grad_x, dx_dt, t, self.nOptIdx_SCAC_in_all)
        return Q_flat, P_flat, G_flat, H_flat, A_flat, B_flat
    
    def GetQP_PT_ACSC(self, x:torch.tensor, t:torch.tensor, dx_dt:torch.tensor):
        """Get Prescibed-time QP optimization solution for state constraint and action constraint

        Args:
            x (torch.tensor): [batch, horizon, act_dim+state_dim]
            t (torch.tensor): scalar
            dx_dt (torch.tensor): [batch, horizon, act_dim+state_dim]
        Returns:
            sol_SCAC (torch.tensor): [batch, horizon, act_dim+state_dim]
        """
        if t<=self.start_time:
            return dx_dt
        
        self.set_device(x.device)
        batch, hor, all_dim = x.shape
        # set CBF
        Q_flat, P_flat, G_flat, H_flat, A_flat, B_flat = self.__constructPT_SCAC_coeff(x, t, dx_dt)
        
        
        #solve QP
        sol_QP_flat = QPFunction(verbose=-1, solver = QPSolvers.PDIPM_BATCHED)(Q_flat, P_flat, G_flat, H_flat, A_flat, B_flat)
        sol_QP = sol_QP_flat.reshape(batch, hor, self.nDimOpt_SCAC)
        
        #get final sol
        sol_SCAC = dx_dt
        sol_SCAC[:,:,self.nOptIdx_SCAC_in_all] = sol_QP[:,:,self.nOptIdx_SCAC_in_SCAC]
        
        return sol_SCAC
    
    def _ComputeCLFCoeff(self, next_s_minus_f:torch.tensor, jac_s:torch.tensor, jac_a:torch.tensor):
        """_summary_

        Args:
            next_s_minus_f (torch.tensor): [batch,hor-1,state_dim]
            jac_s (torch.tensor): [batch,hor-1,state_dim,state_dim]
            jac_a (torch.tensor): [batch,hor-1,state_dim,act_dim]

        Returns:
            G_S_forD (torch.tensor): [batch, hor−1, state_dim]
            G_A_forD (torch.tensor): [batch, hor−1, act_dim]
            G_nextS_forD (torch.tensor): [batch, hor−1, state_dim]
        """
        G_S_forD = torch.einsum('bts,btse->bte', next_s_minus_f, -jac_s) # [batch, hor−1, state_dim]
        G_A_forD = torch.einsum('bts,btse->bte', next_s_minus_f, -jac_a) # [batch, hor−1, act_dim]
        G_nextS_forD = next_s_minus_f # [batch, hor−1, state_dim]
            
        return G_S_forD, G_A_forD, G_nextS_forD
    
    def GetQP_PT_ACSCDC(self, x:torch.tensor, t:torch.tensor, dx_dt:torch.tensor, ACSCDCFlag:int=1):
        """Get Prescibed-time QP optimization solution for state, action constraint, and dynamic constraint

        Args:
            x (torch.tensor): [batch, horizon, act_dim+state_dim]
            t (torch.tensor): scalar
            dx_dt (torch.tensor): [batch, horizon, act_dim+state_dim]
        Returns:
            sol_ACSCDC (torch.tensor): [batch, horizon, act_dim+state_dim]
        """
        
        # early return no CBF/CLF
        if t<=self.start_time:
            return dx_dt

        self.set_device(x.device)
        batch, hor, _ = x.shape
        action = x[:,:-1,:self.nDim_action] # batch, hor-1, act_dim
        state = x[:,:-1,self.nDim_action:] # batch, hor-1, state_dim
        state_next = x[:,1:,self.nDim_action:]# batch, hor-1, state_dim
        
        sol_QP = self.__Compute_ACSCDC_AllInOne(hor, batch, x, t, dx_dt,
                                                  state_next, state, action)

        return sol_QP
    
    def compute_dc_column_indices(
        self,
        horizon: int,
        action_dim: int,
        state_dim: int,
        all_dim: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Pre-compute ALL DC column indices at once.
        
        DC constraint structure:
            G[DC, :] = [G_A_forD | G_S_forD | G_NextS_forD | G_A_forD | ...]
            Each set of (A, S, NextS) appears once per timestep
            
        Args:
            horizon: T (horizon length)
            action_dim: Dimension of action
            state_dim: Dimension of state  
            all_dim: action_dim + state_dim
            
        Returns:
            Dictionary with pre-computed indices for all DC timesteps
        """
        
        # ===== INITIAL ASSIGNMENT (timestep 0) =====
        # Structure: [a0, | a1, s1, | a2, s2, | ...]
        # Initial: a0 at [0, action_dim), next_s(s1) at [2*action_dim, 2*action_dim+state_dim)
        col_0_ac_start = 0
        col_0_ac_end = action_dim
        col_0_next_s_start = 2 * action_dim
        col_0_next_s_end = col_0_next_s_start + state_dim
        
        # ===== LOOP ITERATIONS (timesteps 1..T-2) =====
        # Loop iteration i (where i=[0,1,...,horizon-3]) processes coefficients for timestep i+1
        # Columns for loop iteration i:
        #   a_{i+1}: action_dim + i*all_dim
        #   s_{i+1}: 2*action_dim + i*all_dim
        #   s_{i+2}: 3*action_dim + state_dim + i*all_dim
        loop_steps = torch.arange(0, horizon - 2, dtype=torch.long)
        
        col_ac_starts = action_dim + loop_steps * all_dim
        col_ac_ends = col_ac_starts + action_dim
        
        col_s_starts = 2 * action_dim + loop_steps * all_dim
        col_s_ends = col_s_starts + state_dim
        
        col_next_s_starts = 3 * action_dim + state_dim + loop_steps * all_dim
        col_next_s_ends = col_next_s_starts + state_dim
        
        return {
            # Initial assignment (timestep 0)
            'col_0_ac_start': col_0_ac_start,
            'col_0_ac_end': col_0_ac_end,
            'col_0_next_s_start': col_0_next_s_start,
            'col_0_next_s_end': col_0_next_s_end,
            
            # Loop iterations (timesteps 1..T-2)
            'n_loop_steps': len(loop_steps),  # Number of loop iterations = horizon - 2
            'col_ac_starts': col_ac_starts,
            'col_ac_ends': col_ac_ends,
            'col_s_starts': col_s_starts,
            'col_s_ends': col_s_ends,
            'col_next_s_starts': col_next_s_starts,
            'col_next_s_ends': col_next_s_ends,
        }
        
    def assign_dc_constraints_vectorized(
        self,
        G_flat: torch.Tensor,          # [batch, n_constraints, n_opt_vars]
        H_flat: torch.Tensor,          # [batch, n_constraints]
        G_a_forD: torch.Tensor,        # [batch, horizon-1, action_dim]
        G_s_forD: torch.Tensor,        # [batch, horizon-1, state_dim]
        G_next_s_forD: torch.Tensor,   # [batch, horizon-1, state_dim]
        H_clf_dc: torch.Tensor,        # [batch, horizon-1]
        dc_indices: Dict,
        dc_row_position: int,          # Where to place DC constraints (0 for OnlyDC, hor*nDim_SCAC-nDim_state for ACSCDC)
    ) -> None:
        """
        UNIFIED DC constraint assignment for both ACSCDC_AllInOne and OnlyDC_AllInOne.
        
        Structure: Accumulate ALL DC constraints into a SINGLE row.
        - Initial assignment (t=0): a0, s1
        - Loop iterations (t=1..T-2): a_i, s_i, s_{i+1}
        
        Args:
            G_flat, H_flat: QP constraint matrices
            G_a_forD, G_s_forD, G_next_s_forD: CLF Jacobian coefficients [batch, horizon-1, ...]
            H_clf_dc: CLF constraint values [batch, horizon-1] (summed across all timesteps)
            dc_indices: Pre-computed indices from compute_dc_column_indices()
            dc_row_position: Row where to place the DC constraint
        """
        
        row_dc = dc_row_position
        
        # ===== INITIAL ASSIGNMENT (t=0) =====
        col_ac_s = dc_indices['col_0_ac_start']
        col_ac_e = dc_indices['col_0_ac_end']
        col_next_s_s = dc_indices['col_0_next_s_start']
        col_next_s_e = dc_indices['col_0_next_s_end']
        
        G_flat[:, row_dc, col_ac_s:col_ac_e] = G_a_forD[:, 0, :]
        G_flat[:, row_dc, col_next_s_s:col_next_s_e] = G_next_s_forD[:, 0, :]
        
        # ===== LOOP ITERATIONS (t=1..T-2) =====
        n_loop_steps = dc_indices['n_loop_steps']
        for i in range(n_loop_steps):
            col_ac_s = dc_indices['col_ac_starts'][i].item()
            col_ac_e = dc_indices['col_ac_ends'][i].item()
            col_s_s = dc_indices['col_s_starts'][i].item()
            col_s_e = dc_indices['col_s_ends'][i].item()
            col_next_s_s = dc_indices['col_next_s_starts'][i].item()
            col_next_s_e = dc_indices['col_next_s_ends'][i].item()
            
            # Accumulate into same row
            G_flat[:, row_dc, col_ac_s:col_ac_e] += G_a_forD[:, i+1, :]
            G_flat[:, row_dc, col_s_s:col_s_e] += G_s_forD[:, i+1, :]
            G_flat[:, row_dc, col_next_s_s:col_next_s_e] += G_next_s_forD[:, i+1, :]
        
        # ===== SET H VALUE =====
        # H accumulates all timesteps
        H_flat[:, row_dc] = H_clf_dc.sum(dim=1)
            
    def compute_cbf_ac_sc_indices(
        self,
        horizon: int,
        action_dim: int,
        n_dim_opt_sc: int,  # Optimizable state dimensions (NOT full state_dim!)
        all_dim: int,
        n_dim_ac: int,
        n_dim_sc: int,
        n_dim_scac: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Pre-compute ALL CBF AC/SC row and column indices.
        
        CRITICAL: SC constraints only apply to optimizable state dimensions (n_dim_opt_sc),
        NOT the full state_dim. For example, if position is constrained but velocity is not,
        n_dim_opt_sc = 2 (x, y) even though state_dim = 4 (x, y, vx, vy).
        
        CBF AC/SC constraint structure:
        - Initial AC (a0): rows [0, n_dim_ac)
        - For each loop iteration i (i=0..T-2):
          * AC for a_{i+1}: rows [n_dim_ac + i*n_dim_scac, n_dim_ac + i*n_dim_scac + n_dim_ac)
          * SC for s_{i+1}: rows [n_dim_ac + i*n_dim_scac + n_dim_ac, n_dim_ac + (i+1)*n_dim_scac)
        
        Args:
            horizon: T (horizon length)
            action_dim: Dimension of action
            n_dim_opt_sc: Dimension of optimizable state (NOT full state_dim!)
            all_dim: action_dim + full_state_dim
            n_dim_ac: Number of action constraints
            n_dim_sc: Number of state constraints
            n_dim_scac: Total AC + SC = n_dim_ac + n_dim_sc
            
        Returns:
            Dictionary with pre-computed row/column indices for all CBF timesteps
        """
        
        # AC constraints at first timestep (rows 0..n_dim_ac-1, cols 0..action_dim-1)
        ac_row_start = 0
        ac_row_end = n_dim_ac
        ac_col_start = 0
        ac_col_end = action_dim
        
        # Loop iteration space: i = 0, 1, ..., horizon-2
        loop_steps_cols = torch.arange(0, horizon - 1, dtype=torch.long)
        
        # AC constraints in loop: a_{i+1} at rows [n_dim_ac + i*n_dim_scac, n_dim_ac + i*n_dim_scac + n_dim_ac)
        ac_row_positions_loop = n_dim_ac + loop_steps_cols * n_dim_scac  # [horizon-1] row start positions
        ac_col_starts_loop = all_dim * loop_steps_cols + action_dim  # [2+6*i, 4+6*i, ...]
        ac_col_ends_loop = ac_col_starts_loop + action_dim
        
        # SC constraints in loop: s_{i+1} at rows [n_dim_ac + i*n_dim_scac + n_dim_ac, n_dim_ac + (i+1)*n_dim_scac)
        sc_row_positions_loop = ac_row_positions_loop + n_dim_ac  # Offset by n_dim_ac from AC
        sc_col_starts_loop = all_dim * loop_steps_cols + 2 * action_dim  # [4+6*i, 10+6*i, ...]
        sc_col_ends_loop = sc_col_starts_loop + n_dim_opt_sc  # Use nDimOpt_SC here!
        
        return {
            'ac_row_start': ac_row_start,
            'ac_row_end': ac_row_end,
            'ac_col_start': ac_col_start,
            'ac_col_end': ac_col_end,
            'ac_row_positions_loop': ac_row_positions_loop,    # [horizon-1] (for loop iterations 0..T-2)
            'ac_col_starts_loop': ac_col_starts_loop,          # [horizon-1] (for loop iterations 0..T-2)
            'ac_col_ends_loop': ac_col_ends_loop,              # [horizon-1] (for loop iterations 0..T-2)
            'sc_row_positions_loop': sc_row_positions_loop,    # [horizon-1] (for loop iterations 0..T-2)
            'sc_col_starts_loop': sc_col_starts_loop,          # [horizon-1] (for loop iterations 0..T-2)
            'sc_col_ends_loop': sc_col_ends_loop,              # [horizon-1] (for loop iterations 0..T-2)
            'n_dim_ac': n_dim_ac,
            'n_dim_sc': n_dim_sc,
        }
        
    def compute_clf_dc(
        self,
        state_next: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        x: torch.Tensor,
        batch: int,
        horizon: int,
        T_gain_clf: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Compute CLF (Dynamics Constraint) values and matrices.
        
        IDENTICAL computation in both ACSCDC and OnlyDC functions.
        
        Args:
            self: Controller instance (has jacobian methods)
            state_next: Next state [batch, horizon-1, state_dim]
            state: Current state [batch, horizon-1, state_dim]
            action: Action [batch, horizon-1, action_dim]
            x: Full trajectory [batch, horizon, state+action]
            batch: Batch size
            horizon: T (horizon length)
            T_gain_clf: Time gain for CLF
            
        Returns:
            (next_s_minus_f, jac_s, jac_a, H_CLF_DC, G_S_forD, G_A_forD, G_nextS_forD, CLF_V_cur)
        """
        
        # Compute jacobians
        if self.b_expJ:
            next_s_minus_f = state_next - self.expForwardModel_normed(state, action)
            jac_s, jac_a = self._get_explicit_jacobian()
        else:
            next_s_minus_f = state_next - self.FD_model(state, action)
            x_flat = x[:, :-1, :].reshape(batch * (horizon - 1), self.nDim_all)
            jac_s, jac_a = self._get_jacobian(x_flat, self.nDim_state, self.nDim_action, batch, horizon - 1)
        
        # Compute energy and CLF constraint value
        V_energy = 0.5 * (next_s_minus_f ** 2).sum(dim=2)
        H_CLF_DC = -V_energy * self.c_PTCLF * T_gain_clf - self.d_robust
        CLF_V_cur = V_energy.sum(-1).mean().item()
        
        # Compute CLF coefficients
        G_S_forD, G_A_forD, G_nextS_forD = self._ComputeCLFCoeff(next_s_minus_f, jac_s, jac_a)
        
        return next_s_minus_f, jac_s, jac_a, H_CLF_DC, CLF_V_cur, G_S_forD, G_A_forD, G_nextS_forD
    
    def compute_and_assign_clf_dc(
        self,
        state_next: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        x: torch.Tensor,
        batch: int,
        horizon: int,
        T_gain_clf: float,
        G_flat: torch.Tensor,
        H_flat: torch.Tensor,
        dc_row_position: int,
    ) -> Tuple[float]:
        """
        Compute CLF values and assign DC constraints (SHARED between ACSCDC and OnlyDC).
        
        This entire sequence is IDENTICAL in both functions. The only difference is dc_row_position:
        - ACSCDC: dc_row_position = hor * nDim_SCAC - nDim_SC (DC comes after AC/SC)
        - OnlyDC:  dc_row_position = 0 (DC starts at first row, no AC/SC before it)
        
        Args:
            state_next, state, action, x, batch, horizon, T_gain_clf: CLF computation inputs
            G_flat, H_flat: QP constraint matrices (modified in-place)
            dc_row_position: Starting row for DC constraints in G_flat/H_flat
            
        Returns:
            CLF_V_cur: Current CLF energy value (used for early exit check)
        """
        
        # ===== STEP 1: Compute CLF values =====
        next_s_minus_f, jac_s, jac_a, H_CLF_DC, CLF_V_cur, G_S_forD, G_A_forD, G_nextS_forD = self.compute_clf_dc(
            state_next, state, action, x, batch, horizon, T_gain_clf
        )
        
        # ===== STEP 2: Assign DC constraints =====
        dc_indices = self.compute_dc_column_indices(horizon, self.nDim_action, self.nDim_state, self.nDim_all)
        
        self.assign_dc_constraints_vectorized(
            G_flat, H_flat,
            G_A_forD, G_S_forD, G_nextS_forD,
            H_CLF_DC,
            dc_indices,
            dc_row_position=dc_row_position,
        )
        
        return CLF_V_cur
    
    def early_exit_and_solve_qp(
        self,
        t: float,
        CLF_V_cur: float,
        x: torch.Tensor,
        dx_dt: torch.Tensor,
        Q_flat: torch.Tensor,
        P_flat: torch.Tensor,
        G_flat: torch.Tensor,
        H_flat: torch.Tensor,
        A_flat: torch.Tensor,
        B_flat: torch.Tensor,
        nOpt_dim_total_horizon: int,
        dc_row_position: int,
        nNumTotalConstraint: int,
        horizon: int,
    ) -> Tuple[bool, torch.Tensor]:
        """
        Check early exit conditions and solve QP.
        
        IDENTICAL logic in both ACSCDC and OnlyDC functions.
        
        Args:
            self: Controller instance (has methods like GetQP_PT_ACSC)
            t: Current time
            CLF_V_cur: Current CLF value
            x: Full trajectory
            dx_dt: Target derivatives
            Q_flat, P_flat, G_flat, H_flat, A_flat, B_flat: QP matrices
            nOpt_dim_total_horizon: Optimization dimension
            dc_row_position: DC constraint row position
            nNumTotalConstraint: Total number of constraints
            horizon: T (horizon length)
            
        Returns:
            (early_exit_flag, sol_QP)
            - early_exit_flag: True if early exit, False if solved
            - sol_QP: Solution trajectory [batch, horizon, state+action]
        """
        
        batch = dx_dt.shape[0]
        nRowDC = torch.arange(dc_row_position, nNumTotalConstraint, 1, device=self.device)
        
        org_flow_without_s1 = torch.zeros([batch, nOpt_dim_total_horizon], device=self.device)
        org_flow_without_s1[:, :self.nDim_action] = dx_dt[:, 0, :self.nDim_action]
        org_flow_without_s1[:, self.nDim_action:] = dx_dt[:, 1:, :].reshape(batch, (horizon - 1) * self.nDim_all)
        
        # Check early exit conditions
        if t <= self.start_time:
            return True, dx_dt  # Early exit flag = True
        elif CLF_V_cur < self.cfg.Stop_V:
            return True, self.GetQP_PT_ACSC(x, t, dx_dt)  # Early exit flag = True
        
        # Solve QP
        sol_QP_flat = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(
            Q_flat, P_flat, G_flat, H_flat, A_flat, B_flat
        )
        
        # Reconstruct solution
        left_sol_QP_flat = sol_QP_flat[:, :self.nDim_action]
        right_sol_QP_flat = sol_QP_flat[:, self.nDim_action:]
        sol_QP_flat_complete = torch.cat(
            [left_sol_QP_flat, dx_dt[:, 0, self.nDim_action:], right_sol_QP_flat],
            dim=1
        )
        sol_QP = sol_QP_flat_complete.reshape(batch, horizon, self.nDim_all)
        
        return False, sol_QP  # Early exit flag = False (solved normally)
    
    def compute_cbf_constraints(
        self,
        x: torch.Tensor,
        horizon: int,
        T_gain_cbf: float,
        c_cbf_sc: float,
        c_cbf_ac: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute all CBF values and matrices.
        
        Args:
            self: CBF controller instance (has __computeSC_H, __computeAC_H, etc.)
            x: State trajectory [batch, horizon, state_dim]
            horizon: T (horizon length)
            T_gain_cbf: Time gain for CBF
            c_cbf_sc: CBF coefficient for state constraints
            c_cbf_ac: CBF coefficient for action constraints
            
        Returns:
            (h_CBF_SC, h_CBF_AC, h_grad_SC, H_CBF_SC, H_CBF_AC)
            where H values are the computed constraint bounds
        """
        
        x_SC_opt = x[:, :, self.nOptIdx_SC_in_all]
        x_AC_opt = x[:, :, self.nOptIdx_AC_in_all]
        
        # Compute CBF h values
        h_CBF_SC = self.__computeSC_H(x_SC_opt)  # [batch, horizon, n_dim_sc]
        h_CBF_AC = self.__computeAC_H(x_AC_opt)  # [batch, horizon, n_dim_ac]
        h_grad_SC = self.__computeSC_HGrad(x_SC_opt)  # [batch, horizon, n_dim_sc, ...]
        
        # Compute CBF H matrices (constraint bounds)
        H_CBF_SC = c_cbf_sc * T_gain_cbf * h_CBF_SC - self.s_robust
        H_CBF_AC = c_cbf_ac * T_gain_cbf * h_CBF_AC - self.a_robust
        
        return h_CBF_SC, h_CBF_AC, h_grad_SC, H_CBF_SC, H_CBF_AC
    
    def assign_cbf_constraints(
        self,
        G_flat: torch.Tensor,
        H_flat: torch.Tensor,
        G_AC: torch.Tensor,
        H_CBF_AC: torch.Tensor,
        h_grad_SC: torch.Tensor,
        H_CBF_SC: torch.Tensor,
        cbf_indices: Dict,
        horizon: int,
        action_dim: int,
        state_dim: int,
        n_dim_scac: int,
    ) -> None:
        """
        Assign all CBF AC/SC constraints to G_flat and H_flat.
        
        Structure:
        - Initial AC (a0): rows [0, n_dim_ac), cols [0, action_dim)
        - For loop iteration i (i=0..horizon-2):
          * AC for a_{i+1}: rows [n_dim_ac + i*n_dim_scac, n_dim_ac + i*n_dim_scac + n_dim_ac)
          * SC for s_{i+1}: rows [n_dim_ac + i*n_dim_scac + n_dim_ac, n_dim_ac + (i+1)*n_dim_scac)
        
        Args:
            G_flat, H_flat: QP matrices to fill
            G_AC: Precomputed AC constraint matrix [n_dim_ac, action_dim]
            H_CBF_AC: AC constraint bounds [batch, horizon, n_dim_ac]
            h_grad_SC: SC constraint gradients [batch, horizon, n_dim_sc, ...]
            H_CBF_SC: SC constraint bounds [batch, horizon, n_dim_sc]
            cbf_indices: Pre-computed indices from compute_cbf_ac_sc_indices()
            horizon: T (horizon length)
            action_dim: Dimension of action
            state_dim: Dimension of state
            n_dim_scac: Total AC + SC
        """
        
        batch = G_flat.shape[0]
        n_dim_ac = cbf_indices['n_dim_ac']
        n_dim_sc = cbf_indices['n_dim_sc']
        
        # ===== STEP 1: Assign AC constraints at t=0 (a0) =====
        ac_row_start = cbf_indices['ac_row_start']
        ac_row_end = cbf_indices['ac_row_end']
        ac_col_start = cbf_indices['ac_col_start']
        ac_col_end = cbf_indices['ac_col_end']
        
        G_flat[:, ac_row_start:ac_row_end, ac_col_start:ac_col_end] = G_AC
        H_flat[:, ac_row_start:ac_row_end] = H_CBF_AC[:, 0, :n_dim_ac]
        
        # ===== STEP 2: Assign AC and SC constraints in loop (i=0..horizon-2) =====
        # Each loop iteration assigns both a_{i+1} and s_{i+1}
        
        n_loop_steps = len(cbf_indices['ac_row_positions_loop'])
        for i in range(n_loop_steps):
            timestep = i + 1  # Convert loop index to timestep
            
            # AC for a_{i+1}
            ac_row_pos = cbf_indices['ac_row_positions_loop'][i].item()
            ac_col_start = cbf_indices['ac_col_starts_loop'][i].item()
            ac_col_end = cbf_indices['ac_col_ends_loop'][i].item()
            G_flat[:, ac_row_pos:ac_row_pos+n_dim_ac, ac_col_start:ac_col_end] = G_AC
            H_flat[:, ac_row_pos:ac_row_pos+n_dim_ac] = H_CBF_AC[:, timestep, :n_dim_ac]
            
            # SC for s_{i+1}
            sc_row_pos = cbf_indices['sc_row_positions_loop'][i].item()
            sc_col_start = cbf_indices['sc_col_starts_loop'][i].item()
            sc_col_end = cbf_indices['sc_col_ends_loop'][i].item()
            G_flat[:, sc_row_pos:sc_row_pos+n_dim_sc, sc_col_start:sc_col_end] = -h_grad_SC[:, timestep, :, :]
            H_flat[:, sc_row_pos:sc_row_pos+n_dim_sc] = H_CBF_SC[:, timestep, :]
            
    def build_qp_matrices(
        self,
        horizon: int,
        action_dim: int,
        state_dim: int,
        dx_dt: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build Q, P, A, B matrices for QP.
        
        Minimizes: 0.5 * x^T Q x + P^T x
        Subject to: G x <= H, A x = B
        
        Args:
            horizon: T (horizon length)
            action_dim: Dimension of action
            state_dim: Dimension of state
            dx_dt: Target trajectory derivatives [batch, horizon, state+action]
            device: PyTorch device
            
        Returns:
            (Q_flat, P_flat, A_flat, B_flat)
        """
        
        batch = dx_dt.shape[0]
        n_opt_dim = horizon * (action_dim + state_dim) - state_dim
        
        # Q is identity (minimize deviation from trajectory)
        I = torch.eye(n_opt_dim, device=device).unsqueeze(0)
        Q_flat = I.expand(batch, n_opt_dim, n_opt_dim)
        
        # P is negative of target trajectory (minimize ||x - flow||)
        org_flow_without_s1 = torch.zeros([batch, n_opt_dim], device=device)
        org_flow_without_s1[:, :action_dim] = dx_dt[:, 0, :action_dim]
        org_flow_without_s1[:, action_dim:] = dx_dt[:, 1:, :].reshape(batch, (horizon - 1) * (action_dim + state_dim))
        P_flat = -org_flow_without_s1
        
        # A, B are dummy (no equality constraints)
        A_flat = self.dummy_constraint.to(device)
        B_flat = self.dummy_constraint.to(device)
        
        return Q_flat, P_flat, A_flat, B_flat, org_flow_without_s1

    def solve_qp_and_construct_solution(
        self,
        Q_flat: torch.Tensor,
        P_flat: torch.Tensor,
        G_flat: torch.Tensor,
        H_flat: torch.Tensor,
        A_flat: torch.Tensor,
        B_flat: torch.Tensor,
        dx_dt: torch.Tensor,
        horizon: int,
        action_dim: int,
        state_dim: int,
        solver=None,  # QPFunction solver
    ) -> torch.Tensor:
        """
        Solve QP and construct solution trajectory.
        
        Args:
            Q_flat, P_flat, G_flat, H_flat, A_flat, B_flat: QP matrices
            dx_dt: Target trajectory [batch, horizon, state+action]
            horizon: T (horizon length)
            action_dim: Dimension of action
            state_dim: Dimension of state
            solver: QPFunction solver (e.g., QPSolvers.PDIPM_BATCHED)
            
        Returns:
            sol_QP: Solution trajectory [batch, horizon, state+action]
        """
        
        batch = dx_dt.shape[0]
        
        # Solve QP
        if solver is None:
            from qpth.qp import QPFunction, QPSolvers
            solver = QPSolvers.PDIPM_BATCHED
        
        sol_QP_flat = QPFunction(verbose=-1, solver=solver)(
            Q_flat, P_flat, G_flat, H_flat, A_flat, B_flat
        )
        
        # Reconstruct solution with first state from dx_dt
        left_sol_QP_flat = sol_QP_flat[:, :action_dim]
        right_sol_QP_flat = sol_QP_flat[:, action_dim:]
        sol_QP_flat_complete = torch.cat(
            [left_sol_QP_flat, dx_dt[:, 0, action_dim:], right_sol_QP_flat],
            dim=1
        )
        sol_QP = sol_QP_flat_complete.reshape(batch, horizon, state_dim + action_dim)
        
        return sol_QP

    def __Compute_ACSCDC_AllInOne(
        self, hor: int, batch: int, x: torch.Tensor, t: float, dx_dt: torch.Tensor,
        state_next: torch.Tensor, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        FULLY REFACTORED version using extracted helper functions.
        
        Structure:
        1. Initialize QP matrices
        2. Compute CBF values (packaged function)
        3. Assign CBF constraints (packaged function)
        4. Compute CLF values
        5. Assign DC constraints (packaged function)
        6. Build and solve QP (packaged function)
        
        Changes from original:
        - ~40 lines for CBF: delegated to compute_cbf_constraints() + assign_cbf_constraints()
        - ~30 lines for DC: delegated to assign_dc_constraints_vectorized()
        - ~30 lines for QP: delegated to build_qp_matrices() + solve_qp_and_construct_solution()
        - Total: 100 lines -> 60 lines of logic
        """
        
        nOpt_dim_total_horizon = hor * self.nDim_all - self.nDim_state
        nNumTotalConstraint = (hor - 1) * self.nDim_SCAC + self.nDim_AC + self.nDim_DC
        G_flat = torch.zeros([batch, nNumTotalConstraint, nOpt_dim_total_horizon], device=self.device)
        H_flat = torch.zeros([batch, nNumTotalConstraint], device=self.device)
        
        T_Gain_CBF = self._PTCBF_TimeGain_CBF(t)
        T_Gain_CLF = self._PTCBF_TimeGain_CLF(t)
        
        # ========================================================================
        # STEP 1: CBF Computation (PACKAGED in compute_cbf_constraints)
        # ========================================================================
        if self.b_cPT_calculated is False:
            self.c_PTCBF_SC = self.cfg.CBF_c[0]
            self.c_PTCBF_AC = self.cfg.CBF_c[1]
            self.b_cPT_calculated = True
        
        h_CBF_SC, h_CBF_AC, h_grad_SC, H_CBF_SC, H_CBF_AC = self.compute_cbf_constraints(
            x, hor, T_Gain_CBF,
            self.c_PTCBF_SC, self.c_PTCBF_AC
        )
        
        # ========================================================================
        # STEP 2: CBF AC/SC Assignment (PACKAGED in assign_cbf_constraints)
        # ========================================================================
        cbf_indices = self.compute_cbf_ac_sc_indices(
            hor,
            self.nDim_action,
            self.nDimOpt_SC,  # SC constraints use optimizable state dims, NOT full state_dim
            self.nDim_all,
            self.nDim_AC,
            self.nDim_SC,
            self.nDim_SCAC,
        )
        
        self.assign_cbf_constraints(
            G_flat, H_flat,
            self.G_AC,
            H_CBF_AC,
            h_grad_SC,
            H_CBF_SC,
            cbf_indices,
            hor,
            self.nDim_action,
            self.nDim_state,
            self.nDim_SCAC,
        )
        
        # ========================================================================
        # STEP 3-4: Compute CLF + Assign DC (SHARED - compute_and_assign_clf_dc)
        # ========================================================================
        dc_row_position = hor * self.nDim_SCAC - self.nDim_SC
        CLF_V_cur = self.compute_and_assign_clf_dc(
            state_next, state, action, x,
            batch, hor, T_Gain_CLF,
            G_flat, H_flat,
            dc_row_position
        )
        
        # ========================================================================
        # STEP 5: Build QP matrices
        # ========================================================================
        Q_flat, P_flat, A_flat, B_flat, org_flow_without_s1 = self.build_qp_matrices(
            hor, self.nDim_action, self.nDim_state, dx_dt, self.device
        )
        
        # ========================================================================
        # STEP 6: Early exit and solve QP (PACKAGED in early_exit_and_solve_qp)
        # ========================================================================
        early_exit_flag, sol_QP = self.early_exit_and_solve_qp(
            t, CLF_V_cur, x, dx_dt,
            Q_flat, P_flat, G_flat, H_flat, A_flat, B_flat,
            nOpt_dim_total_horizon, dc_row_position, nNumTotalConstraint,
            hor
        )
        
        return sol_QP