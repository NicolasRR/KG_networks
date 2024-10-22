class MountainCarEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """Rest of the environment definition omitted."""

    def compute_observations(self):
        position, velocity = self.state
        ...
        # position is the position of the car on the x axis
        # velocity is the velocity of the car
        ...
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        ...
        self.state = position, velocity 

        return self.state