import torch
import random
import traceback
import numpy as np
from collections import deque
from BouncingBall import BouncingBallAI, Direction
from model import Linear_Qnet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
Lr = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_Qnet(9, 256, 3)
        self.trainer = QTrainer(self.model, lr=Lr, gamma=self.gamma)

    def get_state(self, game): #get all states like danger right, up, down, left
       

        state = [
            
            game.platform_pos[1] / 600,
            game.ball_pos[0] / 800,
            game.ball_pos[1] / 600,
            
            (game.ball_pos[0] - game.platform_pos[0]) / 800,  # Relative X position
            (game.ball_pos[1] - game.platform_pos[1]) / 600,  # Relative Y position
    
            # game.ball_pos[0] - game.platform_pos[0],  # Relative X position
            # game.ball_pos[1] - game.platform_pos[1],  # Relative Y position
            
            game.SPEED[0] / 10,  # Assuming max velocity ~10
            game.SPEED[1] / 10,
        
            # #ball and platform position
            # game.ball_pos[0],
            # game.ball_pos[1],
            # game.platform_pos[0],
            
            # # #speed/velocity of ball
            # game.SPEED[0],
            # game.SPEED[1],
            
            #ball direction             
            1 if game.SPEED[0] > 0 else 0, #right moving >0 else left
            1 if game.SPEED[1] > 0 else 0 #Down mobving >0 else up
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  #MAX MEMO REACH POPLEFT --> one tupel save

    def train_long_memory(self): #take past samples 
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
    def train_short_memory(self, state, action, reward, next_state, done):#take previous samples 
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #randome moves: trade off exploration --> at start to explore enviroment / exploitation --> after some gmaes it will be better
        self.epsilon = max(0, 80 - self.n_games) 
        final_move = [0,0,0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            
        final_move[move] = 1

        return final_move

def train():
    try:
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        agent = Agent()
        game = BouncingBallAI()
        
        while True:
            # get old state
            state_old = agent.get_state(game) #ballpos, platformpos , ballvelocity, balldirection

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            
            #get new state
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                print("game over")
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                print("plot chart")
                plot(plot_scores, plot_mean_scores)
                
    except Exception as e:
        print(f"An error occurred from train:- {e}");
        tb = traceback.print_exc()
        print(tb)

if __name__ == '__main__':
    train()