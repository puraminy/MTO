import numpy as np
import torch
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
def an(temperature, step, router, amin=1., arate=0.001):
     temperature = max(amin, temperature * np.exp(arate * step))
     router = RelaxedBernoulli(temperature=temperature, 
            logits=router).rsample()            
     return router, temperature 

class A():
    anneal_min = 0.001
    anneal_dir = -1

    def anneal(self, i_step):
         t = max(self.anneal_min,
             self.router_temperature * np.exp(self.anneal_dir * self.anneal_rate * i_step))
         self.router_temperature = t

