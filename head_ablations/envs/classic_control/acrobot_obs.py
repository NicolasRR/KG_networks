class AcrobotEnv(core.Env):
    """Rest of the environment definition omitted."""

    def _get_ob(self):
        s = self.state
        # s[0], s[2]: angle and angular velocity of the first joint (angle 0 = first link points downward)
        # s[1]: angle between the two links (0 = same angle)
        # s[3]: angular velocity of the second joint

        return np.array(
            [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]])