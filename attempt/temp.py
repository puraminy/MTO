
class Anneal:
    def __init__(self, temperature, anneal_dir, anneal_rate, anneal_min, anneal_type="exp"):
        self.anneal_min = anneal_min
        self.cur_step = 0
        self.temperature = temperature
        self.anneal_dir = anneal_dir
        self.anneal_rate = anneal_rate
        self.anneal_type = anneal_type

    def __iter__(self):
        return self

    def anneal(self, i_step):
         exp_rate = np.exp(self.anneal_dir * self.anneal_rate * i_step)
         t = max(self.anneal_min, self.temperature * exp_rate)
         self.temperature = t
         return t

    def __next__(self):
        if self.temperature < self.anneal_min:
            return self.anneal_min
        else:
            self.cur_step += 1 
            value = self.anneal(self.cur_step)
            return value

