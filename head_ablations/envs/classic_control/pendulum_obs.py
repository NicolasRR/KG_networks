class PendulumEnv(gym.Env):    
    """Rest of the environment definition omitted."""
    def _get_obs(self):
        theta, thetadot = self.state
        # theta is the angle of the pendulum
        # thetadot is the angular velocity of the pendulum
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)