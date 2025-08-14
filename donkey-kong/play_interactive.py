#!/usr/bin/env python3
import gymnasium as gym
import ale_py
import pygame
import sys

# Register ALE environments
gym.register_envs(ale_py)

def play_donkey_kong_interactive():
    # Initialize pygame for keyboard input
    pygame.init()
    
    # Create environment
    env = gym.make("ALE/DonkeyKong-v5", render_mode="human")
    
    print("Starting Interactive Donkey Kong!")
    print("Controls:")
    print("Arrow Keys: Move Mario")
    print("Spacebar: Jump/Fire")
    print("Q: Quit")
    print("R: Reset game")
    
    # Key mappings
    key_to_action = {
        pygame.K_SPACE: 1,      # FIRE (jump)
        pygame.K_UP: 2,         # UP
        pygame.K_RIGHT: 3,      # RIGHT
        pygame.K_LEFT: 4,       # LEFT
        pygame.K_DOWN: 5,       # DOWN
    }
    
    obs, info = env.reset()
    clock = pygame.time.Clock()
    
    try:
        total_reward = 0
        running = True
        
        while running:
            action = 0  # Default: no action
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        obs, info = env.reset()
                        total_reward = 0
                        print("Game reset!")
            
            # Get currently pressed keys
            keys = pygame.key.get_pressed()
            
            # Determine action based on pressed keys
            for key, act in key_to_action.items():
                if keys[key]:
                    action = act
                    break
            
            # Handle combination moves
            if keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
                action = 6  # UP-RIGHT
            elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
                action = 7  # UP-LEFT
            elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
                action = 8  # DOWN-RIGHT
            elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
                action = 9  # DOWN-LEFT
            elif keys[pygame.K_UP] and keys[pygame.K_SPACE]:
                action = 10  # UP-FIRE
            elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
                action = 11  # RIGHT-FIRE
            elif keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
                action = 12  # LEFT-FIRE
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if reward > 0:
                print(f"Score! Reward: +{reward}, Total: {total_reward}")
            
            if terminated or truncated:
                print(f"Game Over! Final Score: {total_reward}")
                print("Press R to restart or Q to quit")
            
            # Control frame rate
            clock.tick(60)  # 60 FPS
            
    except KeyboardInterrupt:
        print("\nGame interrupted")
    finally:
        env.close()
        pygame.quit()
        print("Game closed")

if __name__ == "__main__":
    play_donkey_kong_interactive()
