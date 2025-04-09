import torch
from encoders.encoders import MatPromptEncoder
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup
)

class Optim:
    def __init__(self, paras, lrs):
        self.opts = []
        for para,lr in zip(paras,lrs):
            opt = AdamW(para, lr=lr, betas=(0.9, 0.999))
            self.opts.append(opt)

    @property
    def opts_num(self):
        return len(self.opts)

    def step(self):
        for opt in self.opts:
            opt.step()

    def zero_grad(self):
        for opt in self.opts:
            opt.zero_grad()

    def state_dict(self):
        ret = {}
        for i, opt in enumerate(self.opts):
            ret['opt'+ str(i)] = opt.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        for i, opt in enumerate(self.opts):
            opt.load_state_dict(state_dict['opt'+ str(i)])

    def cuda(self):
        for opt in self.opts:
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

class Scheduler:
    def __init__(self, optim, steps=10000, gamma=.1):
        self.schedulers = []
        for opt in optim.opts:
            self.schedulers.append(torch.optim.lr_scheduler.StepLR(opt, steps, gamma))

    def get_last_lr(self):
        self._last_lr = []
        for s in self.schedulers:
            self._last_lr.append(s.get_last_lr())
        return self._last_lr

    def step(self):
        for sch in self.schedulers:
           sch.step()

    def state_dict(self):
        ret = {}
        for i, sch in enumerate(self.schedulers):
            ret['sch'+ str(i)] = sch.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        for i, sch in enumerate(self.schedulers):
            sch.load_state_dict(state_dict['sch'+ str(i)])

def get_optimizer(model, steps, prompt_lr, router_lr, Az_lr):
    paras = []
    lrs = []
    if model.encoder.router.requires_grad:
        paras.append([model.encoder.router])
        lrs.append(router_lr)
    for encoder in model.prompt_encoders:
        para_list =[p for p in encoder.parameters() if p.requires_grad]
        if para_list:
            paras.append(para_list)
            lrs.append(prompt_lr)
    optimizer = Optim(paras, lrs)
    scheduler = Scheduler(optimizer, steps = steps // optimizer.opts_num)

    return optimizer, scheduler

