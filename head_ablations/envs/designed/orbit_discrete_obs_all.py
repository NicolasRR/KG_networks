class OrbitEnv(core.Env):
    def __init__(self, state=None, lower_bound=5, upper_bound=5.2, initial_vel = 0.0, max_bound=15, respawn_random = False, polar = False, initial_deviation = 1.0):
        # Constants
        self.G = 6.67430e-11        # Gravitational constant (m^3 kg^-1 s^-2)
        self.M = 5.972e24           # Mass of the planet (Earth) in kg
        self.R = _R_         # Radius of the planet (Earth) in meters

        # Simulation Parameters
        self.time_step = 100          # Time step in seconds

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.initial_vel = initial_vel
        self.max_bound = max_bound
        self.respawn_random = respawn_random
        self.initial_deviation = initial_deviation
        
        if state is None:
            position = np.array([(self.lower_bound+self.upper_bound)*self.R/2+self.R*self.initial_deviation, 0.0])
            o_tan = np.array([-position[1],position[0]])/np.linalg.norm(position)
            velocity = o_tan*np.sqrt(self.G*self.M/np.linalg.norm(position))*self.initial_vel

            self.state = np.concatenate((position, velocity))
        self.action_space = spaces.Discrete(5)
        self.polar = polar



    def step(self, action):

        if action == 0:
            action = np.array([1,0])
        elif action == 1:
            action = np.array([0, 1])
        elif action == 2:
            action = np.array([-1,0])
        elif action == 3:
            action = np.array([0,-1])
        elif action == 4:
            action = np.array([0,0])

        position = self.state[0:2]
        velocity = self.state[2:4]
        info = {}
        r = np.linalg.norm(position)
        # Compute gravitational acceleration
        gravity_acc = -self.G * self.M * position / r**3
        info['gravity_acc'] = gravity_acc
        info['r'] = r
        info["action"] = action
        # Get agent's own acceleration
        total_acc = gravity_acc + action

        velocity += total_acc * self.time_step
        position += velocity * self.time_step
        r = np.linalg.norm(position)

        self.state = np.concatenate((position, velocity))

        if r < (self.upper_bound*self.R) and r > (self.lower_bound*self.R):
            reward = 1
        else:
            reward = 0

        # Check for collision with the planet
        if r <= self.R:
            logging.debug(f"Collision with the planet")
            terminated = True
        elif r > self.max_bound*self.R:
            logging.debug(f"Agent has escaped the planet's gravity")
            terminated = True
        else:
            terminated = False
        
        info = {"env_reward": reward} 

        try:
            aux_reward, aux_info = self.compute_reward()
            info["aux_reward"]= aux_reward
            info.update(aux_info)
            reward += aux_reward
        except NotImplementedError:
            pass 
        
        if self.polar:
            x, y = position
            v_x, v_y = velocity
            
            # Compute polar position coordinates
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)  # atan2 gives the correct quadrant
            
            # Compute polar velocity components
            v_r = (x * v_x + y * v_y) / r
            v_theta = (x * v_y - y * v_x) / r
            
            # Append the polar data (r, theta, v_r, v_theta)
            state = np.array([r, theta, v_r, v_theta])
        else:
            state = self.state

        return state, reward, terminated, False, info


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        if self.respawn_random:
            position = np.random.uniform(low=-1, high=1, size=2)
            position *= (self.lower_bound+self.upper_bound)*self.R/(2*np.linalg.norm(position))
        else:
            position = np.array([(self.lower_bound+self.upper_bound)*self.R/2, 0.0])
        position += self.R*self.initial_deviation

        o_tan = np.array([-position[1],position[0]])/np.linalg.norm(position)
        velocity = o_tan*np.sqrt(self.G*self.M/np.linalg.norm(position))*self.initial_vel

        self.state = np.concatenate((position, velocity))
        return self.state, {}

        
    def compute_reward(self):
        raise NotImplementedError