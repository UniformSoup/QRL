import matplotlib.animation as anim
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def play(env, agent, reward_function, state_bounds, max_steps):
    states = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    actions = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    new_states = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    dones = tf.TensorArray(tf.bool, size=0, dynamic_size=True)

    step = 0
    done = False
    observation = env.reset()
    while not done and step < max_steps:
        observation = observation / state_bounds
        action = agent.act(observation).numpy()

        new_observation, reward, done, _  = env.step(action)
        reward = reward_function(reward, observation)
        
        states = states.write(step, observation)
        actions = actions.write(step, action)
        rewards = rewards.write(step, tf.cast(reward, dtype=tf.float32))
        new_states = new_states.write(step, new_observation / state_bounds)
        dones = dones.write(step, done)
        
        observation = new_observation
        step += 1

    return states.stack(), actions.stack(), rewards.stack(), new_states.stack(), dones.stack(), step

def train_agent(env, agent, reward_function, max_steps, max_episodes, state_bounds, success_criterion=(475, 100), progress=False):
    total_rewards = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    episode_lengths = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    episode_threshold, episode_average = success_criterion

    for episode in tf.range(max_episodes):
        states, actions, rewards, new_states, dones, step = play(env, agent, reward_function, state_bounds, max_steps)
        agent.learn(states, actions, rewards, new_states, dones)
        
        total_rewards = total_rewards.write(episode, tf.reduce_sum(rewards))
        episode_lengths = episode_lengths.write(episode, step)
        average_episode_length = np.mean(episode_lengths.stack()[-episode_average:])
        
        if episode >= episode_average and average_episode_length > episode_threshold:
            break
        
        if progress:
            print(f"\rEpisode: {episode}, Steps: {step: >3}, Average Episode Length: {average_episode_length: >4.1f}", end='')
    
    if progress:
        print(f"\nSolved at Episode {episode}, Average Episode Length: {average_episode_length:.1f}\n")
    
    return agent, total_rewards.stack(), episode_lengths.stack()

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