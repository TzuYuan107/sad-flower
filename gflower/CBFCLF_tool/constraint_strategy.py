from abc import ABC, abstractmethod
import torch
from gflower.CBFCLF_tool.CBF_CLF_generator import OpenMazeSystem
from gflower.config.flow_matching import FlowMatchingEvaluationConfig

class ConstraintStrategy(ABC):
    def __init__(self, start_time, presribed_time: float, cfg:FlowMatchingEvaluationConfig):
        super().__init__()
        self.start_time = start_time
        self.prescibed_time = presribed_time
        self.cfg = cfg
    @abstractmethod
    def solve(self, x:torch.tensor, t:torch.tensor, dx_dt:torch.tensor, ACSCDCFlag:int = 1):
        pass
    
    def reset_constant_calculated(self):
        pass
    
    def set_action_limit(self, neg_lim, pos_lim):
        pass
    
    def set_state_limit(self, obs_center, obs_radius):
        pass
    
class MazeStrategy(ConstraintStrategy):
    def __init__(self, start_time, prescibed_time, cfg:FlowMatchingEvaluationConfig,
                 FD_model=None):
        super().__init__(start_time, prescibed_time, cfg)
        self.function_system = OpenMazeSystem(self.start_time, self.prescibed_time, cfg,
                                            FD_model)
        
    @abstractmethod
    def solve(self, x:torch.tensor, t:torch.tensor, dx_dt:torch.tensor, ACSCDCFlag:int = 1):
        pass
    
    def reset_constant_calculated(self):
        self.function_system.ResetConstantCalculated()
        
    def set_action_limit(self, neg_lim, pos_lim):
        self.function_system.set_action_limit(neg_lim, pos_lim)
        
    def set_state_limit(self, obs_center, obs_radius):
        self.function_system.set_state_limit(obs_center, obs_radius)
    
class QP_PT_ACSCDC(MazeStrategy):
    def __init__(self, start_time, prescibed_time, cfg:FlowMatchingEvaluationConfig,
                 FD_model=None):
        super().__init__(start_time, prescibed_time, cfg, FD_model)
        
    def solve(self, x:torch.tensor, t:torch.tensor, dx_dt:torch.tensor, ACSCDCFlag:int = 1):
        return self.function_system.GetQP_PT_ACSCDC(x, t, dx_dt, ACSCDCFlag)

        