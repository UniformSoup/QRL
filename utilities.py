import matplotlib.animation as anim
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def train_agent(env, agent, reward_function, max_steps, max_episodes, state_bounds, success_criterion=(475, 100), progress=False):
    total_rewards = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    episode_lengths = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    episode_threshold, episode_average = success_criterion
    success = False

    for episode in tf.range(max_episodes):
        total_reward, step = agent.train(env, reward_function, state_bounds, max_steps)
        
        total_rewards = total_rewards.write(episode, total_reward)
        episode_lengths = episode_lengths.write(episode, step)
        average_episode_length = np.mean(episode_lengths.stack()[-episode_average:])
        
        if episode >= episode_average and average_episode_length > episode_threshold:
            success = True
            break
        
        if progress:
            print(f"\rEpisode: {episode}, Steps: {step: >3}, Average Episode Length: {average_episode_length: >4.1f}", end='')

    average_episode_length = np.mean(episode_lengths.stack()[-episode_average:])
    if progress and average_episode_length > episode_threshold:
        print(f"\nSolved at Episode {episode}, Average Episode Length: {average_episode_length:.1f}\n")
    
    return total_rewards.stack(), episode_lengths.stack(), success

def render(env, action_fn, max_steps):
    frames = []
    observation = env.reset()
    for i in range(max_steps + 1):
        frames.append(env.render(mode='rgb_array'))
        action = action_fn(env, observation)
        observation, _, done, _ = env.step(action)
        if done: break
    
    return frames

def animation(frames, duration):
    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(frames[0])
    text = ax.text(.05, .95, '0', ha='left',
                   va='top', transform=ax.transAxes)
    fig.tight_layout()

    def draw_frame(frame):
        im.set_data(frames[frame])
        text.set_text(str(frame))
        return im, text
        
    animation = anim.FuncAnimation(fig, draw_frame, interval=duration/len(frames), frames=len(frames), blit=True)
    plt.close(fig)
    
    return animation