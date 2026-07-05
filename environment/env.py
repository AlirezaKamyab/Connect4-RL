import numpy as np
import gymnasium as gym
from renderer import ConnectFourRenderer
from typing import Optional
import pygame


class ConnectFour(gym.Env):
    PLAYER1, PLAYER2 = 1, 2
    def __init__(
            self, 
            ncols:int=7, 
            nrows:int=6, 
            connect:int=4, 
            random_init:bool=False, 
            render_mode:str=None,
            **kwargs
        ):
        super(ConnectFour, self).__init__()

        self.ncols = ncols
        self.nrows = nrows
        self.connect = connect
        self.random_init = random_init
        self.state = None
        self.render_mode = render_mode

        self.observation_space = gym.spaces.Dict({
            'state':gym.spaces.Box(low=0, high=2, shape=(self.nrows, self.ncols), dtype=int)
        })

        self.action_space = gym.spaces.Discrete(ncols)
        self.terminated = False
        if self.render_mode in ['human', 'rgb_array']:
            screen_width = kwargs.get('screen_width', 800)
            screen_height = kwargs.get('screen_height', 600)
            voffset = kwargs.get('voffset', 50)
            self.renderer = ConnectFourRenderer(
                state=self.state, 
                screen_width=screen_width, 
                screen_height=screen_height,
                voffset=voffset
            )

        self.window = None
        self.clock = None
        self.fps = kwargs.get('fps', 60)
        self.__text_board = ""


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.random_init:
            self.state = self.observation_space.sample()['state']
        else:
            self.state = np.zeros(shape=(self.nrows, self.ncols), dtype=np.int32)

        self.terminated = False
        
        if self.render_mode in ['human', 'rgb_array']:
            self.renderer.update_state(self.state)
        
        if self.render_mode == 'human':
            self._render_frame()

        return self.state
    
    def is_valid_action(self, action):
        next_row = np.sum(self.state[:, action] > 0)
        next_row = self.nrows - next_row - 1
        if next_row >= 0:
            return True
        return False
    
    def get_valid_actions(self):
        actions = []
        for a in range(self.ncols):
            if self.is_valid_action(a):
                actions.append(a)
        return actions

    def step(self, action, player:int=1):
        if self.terminated:
            return self.state, (0, 0), self.terminated
        if self.state is None:
            raise ValueError("Please reset the environment first")
        if action < 0 or action >= self.ncols:
            raise ValueError("Invalid action has been taken")
        
        next_row = np.sum(self.state[:, action] > 0)
        next_row = self.nrows - next_row - 1
        if next_row < 0:
            return self.state, (0, 0), self.terminated
        
        self.state[next_row][action] = player
        
        if self.render_mode in ['human', 'rgb_array']:
            self.renderer.update_state(self.state)
            self._render_frame()

        self.terminated, winner = self.check_termination()

        if winner == ConnectFour.PLAYER1:
            player1_reward = 1
            player2_reward = -1
        elif winner == ConnectFour.PLAYER2:
            player2_reward = 1
            player1_reward = -1
        else:
            player1_reward = 0
            player2_reward = 0

        return self.state, (player1_reward, player2_reward), self.terminated
    

    def check_termination(self):
        winner = self.check_winner()
        if winner != 0:
            return True, winner
        
        if np.sum(self.state == 0) == 0:
            return True, winner
        
        if len(self.get_valid_actions()) == 0:
            return True, 0
        
        return False, winner
    

    def check_winner(self):
        for i in range(self.nrows):
            for j in range(self.ncols):
                current_player = self.state[i][j]
                if current_player == 0:
                    continue

                # Check Rows (Vertically)
                winner = True
                for c in range(1, self.connect):
                    if i + self.connect > self.nrows or self.state[i + c][j] != current_player:
                        winner = False
                        break
                if winner:
                    return current_player
                
                # Check Columns (Horizontally)
                winner = True
                for c in range(1, self.connect):
                    if j + self.connect > self.ncols or self.state[i][j + c] != current_player:
                        winner = False
                        break
                if winner:
                    return current_player
                
                # Check Diagonal (Right-Down)
                winner = True
                if i + self.connect <= self.nrows and j + self.connect <= self.ncols:
                    for c in range(1, self.connect):
                        if self.state[i + c][j + c] != current_player:
                            winner = False
                            break
                else:
                    winner = False
                
                if winner:
                    return current_player
                
                # Check Diagonal (Left-Down)
                winner = True
                if i + self.connect <= self.nrows and j - self.connect >= -1:
                    for c in range(1, self.connect):
                        if self.state[i + c][j - c] != current_player:
                            winner = False
                            break
                else:
                    winner = False
                
                if winner:
                    return current_player

                
        return 0
    
    def _render_frame(self):
        if self.render_mode not in ['human', 'rgb_array']:
            return None
        
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.renderer.screen_width, self.renderer.screen_height))
            pygame.display.set_caption('ConnectFour')
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        self.renderer.draw()
        self.render_text()

        if self.render_mode == 'human':
            self.window.blit(self.renderer.screen, self.renderer.screen.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.fps)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.renderer.screen)), axes=(1, 0, 2)
            )

    def update_text(self, text:str):
        self.__text_board = text
        
    def render_text(self):
        font = pygame.font.Font(pygame.font.get_default_font(), 30)
        text = font.render(self.__text_board, True, (0, 0, 0))
        rect = text.get_rect(center=(
            self.renderer.screen_width // 2, self.renderer.voffset // 2
        ))
        self.renderer.screen.blit(text, rect)

    def render(self):
        # Implement rendering logic here
        if self.render_mode == 'rgb_array':
            return self._render_frame()
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
