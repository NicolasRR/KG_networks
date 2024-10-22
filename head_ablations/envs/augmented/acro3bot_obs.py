class acro3bot(BaseEnv):
    """Rest of the environment definition omitted."""

    def get_obs(self):
        return np.array([np.cos(self.state[0]),np.sin(self.state[0]),
                        np.cos(self.state[1]),np.sin(self.state[1]),
                        np.cos(self.state[2]),np.sin(self.state[2]),
                        self.state[3],
                        self.state[4],
                        self.state[5]
                        ])