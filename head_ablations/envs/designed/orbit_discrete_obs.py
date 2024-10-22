class OrbitEnv(core.Env):
    def __init__(self, state=None, lower_bound=5, upper_bound=5.2, extrakwargs={}):
        """
        rest of the environment definition omitted
        """ 
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        position = np.array([(self.lower_bound+self.upper_bound)*self.R/2, 0.0])
        velocity = ...

        self.state = np.array([position[0], position[1], velocity[0], velocity[1]], dtype=np.float32)
