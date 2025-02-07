import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import sys

pygame.init()
#font = pygame.font.Font('arial.ttf', 25)
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
BALL_RADIUS = 20
         
class BouncingBallAI:
    def __init__(self, w=800, h=600):
        self.WIDTH = w
        self.HEIGHT = h
        self.display = pygame.display.set_mode((self.WIDTH,self.HEIGHT))
        pygame.display.set_caption('Bouncing Ball')
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        self.ball_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.platform_pos = [self.WIDTH // 2 - 100 // 2, self.HEIGHT - 10 - 10]
        self.score = 0
        self.game_iter = 0
        self.direction = Direction.RIGHT
        self.SPEED = [5,5]
    
    def play_step(self, action):
        self.game_iter += 1
        reward = 0
        game_over = False;
        
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
    
        self.ball_pos[0] += self.SPEED[0]
        self.ball_pos[1] += self.SPEED[1]
      
        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_LEFT]:
        #     self.platform_pos[0] -=  10
        # elif keys[pygame.K_RIGHT]:
        #     self.platform_pos[0] += 10
        
        if np.array_equal(action, [1, 0, 0]):
            self.direction  = Direction.LEFT
        elif np.array_equal(action, [0, 0,1]):
            self.direction  = Direction.RIGHT
        elif np.array_equal(action, [0, 1,0]): 
            self.direction  = None
            
        if self.direction is None:
            self.platform_pos[0] = self.platform_pos[0]
        elif self.direction == Direction.LEFT:
            self.platform_pos[0] -=  10
        elif self.direction == Direction.RIGHT:
            self.platform_pos[0] += 10 
                        
        self.platform_pos[0] = max(0,min(self.platform_pos[0], self.WIDTH  - 100))
        self.platform_pos[1] = max(0,min(self.platform_pos[1], self.HEIGHT - 10))
        
        if (
            self.platform_pos[0] <= self.ball_pos[0] <= self.platform_pos[0] + 100
            and self.platform_pos[1] -10 <= self.ball_pos[1] <= self.platform_pos[1] + 10
         ):                                                                                                                                                                                                                                                                                                              
            self.SPEED[1] = -self.SPEED[1]
            self.score += 1
            reward += 20
        else:
            reward -= 0.5  #for unecessary move
        
        # if ball goes out of screen then change direction of movement
        if self.ball_pos[0] <= 0 or self.ball_pos[0] >= self.WIDTH:
            self.SPEED[0] = -self.SPEED[0]

        if self.ball_pos[1] <= 0:
            self.SPEED[1] = -self.SPEED[1]
        
        if self.ball_pos[1] > self.HEIGHT:
            self.ball_pos = [self.WIDTH // 2, self.HEIGHT // 2]
            reward -= 10
            game_over = True
            #game_over_screen(score)
            
        self.display.fill(WHITE)
        
        pygame.draw.circle(surface=self.display,color=RED,center=[self.ball_pos[0] ,self.ball_pos[1]] ,radius=BALL_RADIUS)
        pygame.draw.rect(self.display, BLACK,pygame.Rect(self.platform_pos[0],self.platform_pos[1], 100,10))
        #pygame.display.update() 
        
        text = font.render("Score: " + str(self.score), True, BLUE1)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
        self.clock.tick(60)
        return  reward, game_over, self.score
    
    def game_over_screen(self,score):
        self.display.fill(WHITE)
        self.show_text_on_screen("Game Over", 50, self.HEIGHT // 3)
        self.show_text_on_screen(f"Your final score: {score}", 30, self.HEIGHT // 2)
        self.show_text_on_screen("Press any key to restart...", 20, self.HEIGHT * 2 // 3)
        pygame.display.flip()
        self.wait_for_key()
        
    def show_text_on_screen(self,text, font_size, y_position):
        font = pygame.font.Font(None, font_size)
        text_render = font.render(text, True, BLUE1)
        text_rect = text_render.get_rect(center=(self.WIDTH // 2, y_position))
        self.display.blit(text_render, text_rect)
        
    def wait_for_key(self):
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    waiting = False    

# if __name__ == "__main__":
    
#     game = BouncingBallAI()
#     while True:
#       score,reward, game_over =  game.play_step()
      
#       if game_over:
#           break
    
#     pygame.quit()