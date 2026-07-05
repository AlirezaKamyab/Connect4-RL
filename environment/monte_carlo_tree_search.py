import numpy as np
import gymnasium as gym
from typing import Union, Any
from env import ConnectFour
from threading import Thread
import pygame
from tqdm import tqdm


class MCTSNode:
    def __init__(
        self,
        state,
        player:int=0,
        is_terminal:bool=False,
        gamma:float=1.0,
    ):
        self.state = state.copy()
        self.is_terminal = is_terminal
        self.children:dict[int, MCTSNode] = {}
        self.rewards:dict[int, float] = {}
        self.value = 0.0
        self.visits = 0
        self.untried_actions = None
        self.gamma = gamma
        self.player = player

    def is_fully_expanded(self, actions:Union[int, tuple, list]):
        if not isinstance(actions, (tuple, list)):
            actions = list(range(actions))
        if self.untried_actions is None:
            self.untried_actions = [x for x in actions]
        return len(self.untried_actions) == 0
    
    def ucb_select(self, c_param:float=1.41) -> Union["MCTSNode", int]:
        best_child = None
        max_v = -np.inf
        chosen_action = None

        for action, child in self.children.items():
            value = self.rewards[action] + self.gamma * (-child.value / child.visits)
            exploration = c_param * np.sqrt(np.log(self.visits) / child.visits)
            value = value + exploration
            if value > max_v:
                best_child = child
                chosen_action = action
                max_v = value
        
        return best_child, chosen_action
    
    def greedy_select(self):
        action_value = {}
        for action, child in self.children.items():
            value = self.rewards[action] - self.gamma * (child.value / child.visits)
            action_value[action] = value
        
        max_value = max(action_value.values())
        candidates = []
        for action, value in action_value.items():
            if value == max_value:
                candidates.append(action)

        assert len(candidates) > 0, "There is no action to choose from!"
        print(f"AI evaluation is {max_value:.3f}")
        print(action_value)
        return np.random.choice(candidates), max_value
    
    def add_child_by_state(
        self, 
        state:Any, 
        action:int, 
        reward:float, 
        terminated:bool,
        player:int
    ) -> "MCTSNode":
        self.rewards[action] = reward
        node = MCTSNode(
            state=state,
            is_terminal=terminated,
            gamma=self.gamma,
            player=player
        )
        self.children[action] = node
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        return node
    
    def add_child_by_node(
        self,
        node:"MCTSNode",
        action:int,
        reward:float
    ):
        self.rewards[action] = reward
        self.children[action] = node
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        return node
    
    def __eq__(self, value:"MCTSNode") -> bool:
        return self.state == value.state
    
    def __repr__(self) -> str:
        return f"state={self.state}, value={self.value}, visits={self.visits}"
    
    def __str__(self) -> str:
        return self.__repr__()


class MCTS:
    def __init__(
        self,
        env:ConnectFour,
        iterations:int,
        rollout_depth:int=100,
        gamma:float = 1.0,
        c_param:float=1.41,
    ):
        self.env = env
        self.c_param = c_param
        self.rollout_depth = rollout_depth
        self.gamma = gamma
        self.iterations = iterations
        self.state_node = {}

    def search(self, initial_state:Union[np.ndarray, MCTSNode], player:int) -> int:
        if not isinstance(initial_state, MCTSNode):
            root = MCTSNode(
                state=initial_state, gamma=self.gamma, player=player
            )
        else: root = initial_state

        for _ in tqdm(range(self.iterations), leave=False):
            node, trajectory = self.select(root)
            value = self.simulate(node)
            self.backup(node, trajectory, value)

        return root.greedy_select()

    def select(self, node:MCTSNode) -> tuple[MCTSNode, list[MCTSNode]]:
        trajectory = [node]
        depth = 0
        while not node.is_terminal and depth < self.rollout_depth:
            actions = self.__get_valid_actions(node.state)
            if not node.is_fully_expanded(actions):
                node = self.expand(node)
                trajectory.append(node)
                return node, trajectory
            else:
                node, _ = node.ucb_select(self.c_param)
                trajectory.append(node)
                depth += 1
        
        return node, trajectory

    def expand(self, node:MCTSNode) -> MCTSNode:
        action = node.untried_actions.pop()
        next_state, reward, terminal = self.__env_step(node.state, action, node.player)
        reward = reward[node.player]
        player = (node.player + 1) % 2
        
        child_node = node.add_child_by_state(
            state=next_state,
            action=action,
            reward=reward,
            terminated=terminal, 
            player=player
        )
        child_node.gamma = self.gamma
        return child_node

    def simulate(self, node:MCTSNode) -> float:
        done = False
        depth = 0
        state = node.state.copy()
        total_reward = 0
        player = node.player

        while not done:
            actions = self.__get_valid_actions(state)
            if len(actions) == 0: return 0.0
            action = np.random.choice(actions)
            state, reward, done = self.__env_step(state, action, player)
            reward = reward[node.player]
            total_reward += (self.gamma ** depth) * reward
            player = (player + 1) % 2
            depth += 1
        return total_reward
        
    def backup(
        self, 
        node:MCTSNode, 
        trajectory:list[MCTSNode],
        reward:float
    ) -> None:
        for node in reversed(trajectory):
            if not node.is_terminal:
                node.value += reward
            else:
                node.value = 0.0
            node.visits += 1
            reward = -self.gamma * reward
        

    def __reset_env(self, state) -> gym.Env:
        self.env.reset()
        self.env.state = state.copy()
        return self.env
    
    def __get_valid_actions(self, state):
        self.env.reset()
        self.env.state = state.copy()
        return self.env.get_valid_actions()
    
    def __env_step(self, state, action, player:int=0) -> tuple[Any, float, bool]:
        env = self.__reset_env(state)
        next_state, reward, terminated = env.step(action, player + 1)
        return next_state, reward, terminated
    

def main():
    pygame.init()
    settings = {
        'screen_width': 1500,
        'screen_height': 900,
        'voffset': 100
    }
    ncols, nrows, connect = 7, 6, 4
    board = ConnectFour(ncols=ncols, nrows=nrows, connect=connect, render_mode='human', **settings)
    board.reset()
    sim_env = ConnectFour(ncols=ncols, nrows=nrows, connect=connect)
    mcts = MCTS(
        env=sim_env,
        rollout_depth=1000,
        iterations=7500,
        gamma=1.0
    )

    running = True
    ai_thinking = False
    thread:Thread = None
    ai_result = []
    player = 1
    ai_player = 1
    human_player = 2
    while running and not board.terminated:
        if player == ai_player:
            if not ai_thinking:
                print("Thinking...")
                ai_result = []
                thread = Thread(target=ai_worker, args=(ai_result, mcts, board.state, player - 1))
                ai_thinking = True
                thread.start()
            elif ai_thinking and thread is not None and not thread.is_alive():
                action, estimation = ai_result[0]
                board.update_text(f"Player {human_player}'s turn | Estimation: {estimation:.3f}")
                _, rewards, _ = board.step(action, player=player)
                player = human_player
                print("Your turn now\n", end='')
                thread = None
                ai_thinking = False
            elif ai_thinking:
                pygame.event.pump()
                pygame.time.wait(500)

        if board.terminated: break
        if player != human_player:
            pygame.event.clear() 
            continue
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                entry_pos = board.renderer.position_to_entry(mouse_pos)
                if entry_pos is not None and not board.terminated:
                    _, col = entry_pos
                    if not board.is_valid_action(col):
                        print("Please input a valid action")
                        continue
                    _, rewards, _ = board.step(col, player=player)
                    player += 1
                    if player == 3: player = 1
                    board.update_text(f"Player {ai_player}'s turn")
                    board._render_frame()
                    print(rewards, 'player turn is', player, board.terminated)
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE] or event.type == pygame.QUIT:
                running = False
            if keys[pygame.K_r]:
                board.reset()


    winner = None
    match(board.check_winner()):
        case 0: winner = "draw"
        case 1: winner = "player 1"
        case 2: winner = "player 2"
        case _: winner = "draw"
    print('The winner is', winner)

    while True:
        pygame.display.set_caption("The winner is " + winner)
        board.update_text("The winner is " + winner)
        board._render_frame()
        pygame.time.wait(500)
        pygame.event.pump()

def ai_worker(result:list, mcts:MCTS, state, player:int):
    result.append(mcts.search(state, player))


if __name__ == '__main__':
    main()
