import math                                                                                   
                                                                                              
import numpy as np                                                                            
from collections import OrderedDict                                                           
from typing import Any, Callable, List, Tuple, Union                                          
                                                                                              
import torch                                                                                  
from einops import rearrange                                                                  
from einops.layers.torch import Rearrange                                                     
from torch import Tensor, nn                                                                  
from torch.autograd import Variable                                                           
                                                                                              
# other bcos modules                                                                          
from bcos.modules.common import DetachableModule                                              
from bcos.modules.bcosconv2d import *                                                         
from bcos.modules.bcoslinear import *                                                         
from bcos.modules.common import *                                                             
from bcos.modules.logitlayer import *                                                         
from bcos.modules import norms
                                                                                              
from utils import trunc_normal_

class BcosDINOHead(nn.Module):                                                                
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256, linear_layer=BcosLinear, act_layer=nn.Identity):
        super().__init__()                                                                    
        nlayers = max(nlayers, 1)                                                             
        if nlayers == 1:                                                                      
            self.mlp = linear_layer(in_dim, bottleneck_dim)                                   
        else:                                                                                 
            layers = [linear_layer(in_dim, hidden_dim)]                                       
            if use_bn:
                layers.append(norms.NoBias(norms.BatchNormUncentered1d)(hidden_dim))
            layers.append(act_layer())                                                        
            for _ in range(nlayers - 2):                                                      
                layers.append(linear_layer(hidden_dim, hidden_dim))                           
                if use_bn:
                    layers.append(norms.NoBias(norms.BatchNormUncentered1d)(hidden_dim))
                layers.append(act_layer())                                                    
            layers.append(linear_layer(hidden_dim, bottleneck_dim))                           
            self.mlp = nn.Sequential(*layers)                                                 
        self.apply(self._init_weights)                                                        
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False)) # this is just linear, should we also make this BCOS?
        self.last_layer.weight_g.data.fill_(1)                                                
        if norm_last_layer:                                                                   
            self.last_layer.weight_g.requires_grad = False                                    
                                                                                              
    def _init_weights(self, m):                                                               
        if isinstance(m, nn.Linear):                                                          
            trunc_normal_(m.weight, std=0.02)                                                 
            if isinstance(m, nn.Linear) and m.bias is not None:                               
                nn.init.constant_(m.bias, 0)                                                  
                                                                                              
    def forward(self, x):                                                                     
        x = self.mlp(x)                                                                       
        x = nn.functional.normalize(x, dim=-1, p=2) # this operation makes it fail the sanity check.
        x = self.last_layer(x)                                                                
        return x

class DINOHead(nn.Module):                                                                    
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()                                                                    
        nlayers = max(nlayers, 1)                                                             
        if nlayers == 1:                                                                      
            self.mlp = nn.Linear(in_dim, bottleneck_dim)                                      
        else:                                                                                 
            layers = [nn.Linear(in_dim, hidden_dim)]                                          
            if use_bn:                                                                        
                layers.append(nn.BatchNorm1d(hidden_dim))                                     
            layers.append(nn.GELU())                                                          
            for _ in range(nlayers - 2):                                                      
                layers.append(nn.Linear(hidden_dim, hidden_dim))                              
                if use_bn:                                                                    
                    layers.append(nn.BatchNorm1d(hidden_dim))                                 
                layers.append(nn.GELU())                                                      
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))                              
            self.mlp = nn.Sequential(*layers)                                                 
        self.apply(self._init_weights)                                                        
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)                                                
        if norm_last_layer:                                                                   
            self.last_layer.weight_g.requires_grad = False                                    
                                                                                              
    def _init_weights(self, m):                                                               
        if isinstance(m, nn.Linear):                                                          
            trunc_normal_(m.weight, std=.02)                                                  
            if isinstance(m, nn.Linear) and m.bias is not None:                               
                nn.init.constant_(m.bias, 0)                                                  
                                                                                              
    def forward(self, x):                                                                     
        x = self.mlp(x)                                                                       
        x = nn.functional.normalize(x, dim=-1, p=2)                                           
        x = self.last_layer(x)                                                                
        return x
