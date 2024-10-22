class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """Rest of the environment definition omitted."""

    def compute_observations(self):
        
        x, x_dot, theta, theta_dot = self.state
        # x is the horizontal position of the cart
        # x_dot is the velocity of the cart
        # theta is the angle of the pole
        # theta_dot is the angular velocity of the pole
        ...
        self.state = x, x_dot, theta, theta_dot 

        return self.state
