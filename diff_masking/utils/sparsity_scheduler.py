class linear_scheduler():
    def __init__(self, cfg, total_iters):
        self.final_factor = cfg.final_factor
        self.total_iters = total_iters
        self.pivot = cfg.pivot
        self.sparsity = cfg.initial_factor
        self.current_iter = 0
        if self.sparsity == self.final_factor:
            self.factor = 0.0
        else:    
            self.factor = (self.final_factor - self.sparsity)/(self.total_iters*(1-self.pivot))
    
    def update(self):
        self.current_iter += 1
        if self.current_iter >= (self.pivot)*self.total_iters:
            self.sparsity += self.factor

            
SCHEDULERS = {
    "linear": linear_scheduler
}