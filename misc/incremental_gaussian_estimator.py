"""
This class implements incremental mean and variance estimation. It is bsaed
on code that is available here: http://www.johndcook.com/blog/standard_deviation/
"""

class IncrementalGaussianEstimator:
    def __init__(self):
        self.n = 0
        self.old_mean = 0
        self.new_mean = 0
        
        self.old_s = 0
        self.new_s = 0

    def add_observation(self, x):
        self.n += 1

        if self.n == 1:
            self.old_mean = x
            self.new_mean = x

        else:
            self.new_mean = self.old_mean + (x - self.old_mean) / self.n
            self.new_s = self.old_s + (x - self.old_mean) * (x - self.new_mean)

            self.old_mean = self.new_mean
            self.old_s = self.new_s
            
    def add_observations(self, x):
        for xi in x:
            self.add_observation(xi)

    def get_mean(self):
        return self.new_mean

    def get_var(self):
        if self.n > 1:
            return self.new_s / (self.n - 1)

        return 0
