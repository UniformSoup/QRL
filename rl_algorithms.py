import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import Huber, MeanSquaredError, huber, mean_squared_error

@tf.function() 
def calculate_returns(rewards, discount_factor):
    size = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=size)
    discounted_sum = tf.zeros([], dtype=tf.float32)
    for i in tf.range(size - 1, -1, -1):
        discounted_sum = rewards[i] + tf.cast(discount_factor * discounted_sum, dtype=tf.float32)
        returns = returns.write(i, discounted_sum)
    
    returns = returns.stack()
    return (returns - tf.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-8)

class Reinforce():
    """A class for training a discrete policy network."""
    def __init__(self, policy, learning_rate, discount_factor):
        self.gamma = discount_factor
        self.policy = policy
        self.optimizer = Adam(learning_rate)#, amsgrad=True)

    @tf.function()  
    def act(self, observation):
        prediction = self.policy(observation[None, :])
        action = tf.random.categorical(tf.math.log(prediction), num_samples=1)
        return tf.squeeze(action)
    
    @tf.function(reduce_retracing=True)
    def learn(self, states, actions, rewards):
        returns = calculate_returns(rewards, self.gamma)
        actions_one_hot = tf.one_hot(actions, self.policy.output_shape[1]) 

        with tf.GradientTape() as tape:
            probabilities = self.policy(states)
            log_probabilities = tf.math.log(tf.reduce_sum(probabilities * actions_one_hot, axis=1))
            loss = -tf.reduce_mean(log_probabilities * returns)
        
        gradients = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))

    def train(self, env, reward_function, state_bounds, max_steps):
        states = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    
        step = 0
        done = False
        observation = env.reset()
        while not done and step < max_steps:
            observation = observation / state_bounds
            action = self.act(observation).numpy()
    
            new_observation, reward, done, _  = env.step(action)
            reward = reward_function(reward, observation)
            
            states = states.write(step, observation)
            actions = actions.write(step, action)
            rewards = rewards.write(step, tf.cast(reward, dtype=tf.float32))
            
            observation = new_observation
            step += 1

        self.learn(states.stack(), actions.stack(), rewards.stack())
    
        return tf.reduce_sum(rewards.stack()), step

class ActorCritic():
    """A class for training a discrete actor critic network."""
    def __init__(self, model, learning_rate, discount_factor, entropy_coefficient, loss_weights, clipnorm):
        self.gamma = discount_factor
        self.entropy_coefficient = entropy_coefficient
        self.optimizer = Adam(learning_rate, amsgrad=True, clipnorm=clipnorm)
        self.model = model
        self.critic_loss = MeanSquaredError()
        self.loss_weights = loss_weights

    @tf.function
    def act(self, observation):
        prediction, _ = self.model(observation[None, :])
        action = tf.random.categorical(tf.math.log(prediction), num_samples=1)
        return tf.squeeze(action)

    @tf.function
    def loss(self, states, actions, returns, actions_one_hot):
        probabilities, values = self.model(states)

        log_probabilities = tf.math.log(tf.reduce_sum(probabilities * actions_one_hot, axis=1) + 1e-8)
        advantages = returns - values
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        
        actor_loss = - tf.reduce_mean(log_probabilities * advantages)
        critic_loss = tf.reduce_mean(mean_squared_error(returns, values))
        entropy = - self.entropy_coefficient * tf.reduce_mean(tf.reduce_sum(probabilities * tf.math.log(probabilities) + 1e-8, axis=1))
        
        return actor_loss * self.loss_weights[0] + critic_loss * self.loss_weights[1] + entropy
        
    @tf.function(reduce_retracing=True)
    def learn(self, states, actions, rewards):
        returns = calculate_returns(rewards, self.gamma)
        actions_one_hot = tf.one_hot(actions, self.model.output_shape[0][1]) 

        with tf.GradientTape() as tape:
            loss = self.loss(states, actions, returns, actions_one_hot)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def train(self, env, reward_function, state_bounds, max_steps):
        states = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    
        step = 0
        done = False
        observation = env.reset()
        while not done and step < max_steps:
            observation = observation / state_bounds
            action = self.act(observation).numpy()
    
            new_observation, reward, done, _  = env.step(action)
            reward = reward_function(reward, observation)
            
            states = states.write(step, observation)
            actions = actions.write(step, action)
            rewards = rewards.write(step, tf.cast(reward, dtype=tf.float32))
            
            observation = new_observation
            step += 1

        self.learn(states.stack(), actions.stack(), rewards.stack())
    
        return tf.reduce_sum(rewards.stack()), step

# class ActorCritic():
#     """A class for training a discrete actor critic network."""
#     def __init__(self, actor, critic, learning_rate, discount_factor, entropy_coefficient):
#         self.gamma = discount_factor
#         self.entropy_coefficient = entropy_coefficient
#         self.actor = actor
#         self.actor_optimizer = Adam(learning_rate[0], amsgrad=True)
#         self.critic = critic
#         self.critic_optimizer = Adam(learning_rate[1], amsgrad=True)
#         self.critic_loss = MeanSquaredError()

#     @tf.function()  
#     def act(self, observation):
#         prediction = self.actor(observation[None, :])
#         action = tf.random.categorical(tf.math.log(prediction), num_samples=1)
#         return tf.squeeze(action)
    
#     @tf.function(reduce_retracing=True)
#     def learn(self, states, actions, rewards, new_states, dones):
#         returns = calculate_returns(rewards, self.gamma)
#         actions_one_hot = tf.one_hot(actions, self.actor.output_shape[1]) 
        
#         # self.actor.fit(states, actions_one_hot, sample_weight=advantages, verbose=False, batch_size=len(self.memory))
#         # self.critic.fit(states, returns, verbose=False, batch_size=len(self.memory)) 

#         with tf.GradientTape() as critic_tape:
#             values = self.critic(states)
#             loss = self.critic_loss(returns, values)
            
#         critic_gradients = critic_tape.gradient(loss, self.critic.trainable_variables)

#         with tf.GradientTape() as actor_tape:
#             probabilities = self.actor(states)
#             log_probabilities = tf.math.log(tf.reduce_sum(probabilities * actions_one_hot, axis=1))
#             advantages = returns - values
#             advantages = (advantages - tf.reduce_mean(advantages)) / tf.math.reduce_std(advantages)
#             entropy = -tf.reduce_sum(probabilities * tf.math.log(probabilities), axis=1)
#             loss = -tf.reduce_mean(log_probabilities * advantages + entropy * self.entropy_coefficient)

#         actor_gradients = actor_tape.gradient(loss, self.actor.trainable_variables)
        
#         self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
#         self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
          
    # def learn(self):
    #     states, actions, rewards, _, _ = zip(*self.memory)
    #     states = tf.stack(states)
    #     #actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    #     actions, rewards = map(tf.convert_to_tensor, [actions, rewards], [tf.int32, tf.float32])
    #     actions_one_hot = tf.one_hot(actions, self.actor.output_shape[1])
    #     returns, advantages = self.calculate(states, rewards, len(self.memory))
    #     self.actor.fit(states, actions_one_hot, sample_weight=advantages, verbose=False, batch_size=len(self.memory))
    #     self.critic.fit(states, returns, verbose=False, batch_size=len(self.memory))
    #     self.memory = []

class PPO():
    """A class for training a discrete PPO network."""
    def __init__(self, model, learning_rate, discount_factor, entropy_coefficient,
                 loss_weights, clipnorm, clip_epsilon=0.2, n_epochs=10):
        self.gamma = discount_factor
        self.entropy_coefficient = entropy_coefficient
        self.optimizer = Adam(learning_rate, amsgrad=True, clipnorm=clipnorm)
        self.model = model
        self.loss_weights = loss_weights
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs

    @tf.function 
    def act(self, observation):
        prediction, _ = self.model(observation[None, :])
        action = tf.random.categorical(tf.math.log(prediction), num_samples=1)
        return tf.squeeze(action)

    @tf.function
    def logits(self, observation):
        prediction, _ = self.actor(observation[None, :])
        return tf.math.log(prediction)

    @tf.function
    def loss(self, states, actions, returns, old_log_probabilities, actions_one_hot):
        probabilities, values = self.model(states)
        
        advantages = returns - values
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        
        log_probabilities = tf.math.log(tf.reduce_sum(probabilities * actions_one_hot, axis=1) + 1e-8)
        ratio = tf.exp(log_probabilities - old_log_probabilities)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        
        actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
        critic_loss = tf.reduce_mean(mean_squared_error(returns, values))
        entropy = -self.entropy_coefficient * tf.reduce_mean(tf.reduce_sum(probabilities * tf.math.log(probabilities + 1e-8), axis=1))
        
        return actor_loss * self.loss_weights[0] + critic_loss * self.loss_weights[1] + entropy
    
    @tf.function(reduce_retracing=True)
    def learn(self, states, actions, rewards, old_log_probabilities):
        returns = calculate_returns(rewards, self.gamma)
        actions_one_hot = tf.one_hot(actions, self.model.output_shape[0][1])

        for _ in range(self.n_epochs):
            with tf.GradientTape() as tape:
                loss = self.loss(states, actions, returns, old_log_probabilities, actions_one_hot)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


    def train(self, env, reward_function, state_bounds, max_steps):
        states = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        log_probs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        step = 0
        done = False
        observation = env.reset()
        while not done and step < max_steps:
            observation = observation / state_bounds
            prediction, _ = self.model(observation[None, :])
            action = tf.squeeze(tf.random.categorical(tf.math.log(prediction), num_samples=1)).numpy()
            log_prob = tf.math.log(prediction[0, action])

            new_observation, reward, done, _ = env.step(action)
            reward = reward_function(reward, observation)
            
            states = states.write(step, observation)
            actions = actions.write(step, action)
            rewards = rewards.write(step, tf.cast(reward, dtype=tf.float32))
            log_probs = log_probs.write(step, log_prob)

            observation = new_observation
            step += 1

        self.learn(states.stack(), actions.stack(), rewards.stack(), log_probs.stack())
    
        return tf.reduce_sum(rewards.stack()), step


# def get_action_and_value(state):
#     logits = policy(state)
#     action = tf.random.categorical(logits, 1)
#     value = value(state)
#     return tf.squeeze(action, axis=-1), tf.squeeze(value, axis=-1)

# def compute_advantages(rewards, values, next_value, dones):
#     advantages = np.zeros_like(rewards)
#     last_advantage = 0
#     for t in reversed(range(len(rewards))):
#         if dones[t]:
#             last_advantage = 0
#         delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
#         advantages[t] = last_advantage = delta + gamma * 0.95 * (1 - dones[t]) * last_advantage
#         next_value = values[t]
#     returns = advantages + values
#     return advantages, returns

# def ppo_update(states, actions, advantages, returns, old_log_probs):
#     for _ in range(update_epochs):
#         with tf.GradientTape() as tape:
#             logits = policy(states)
#             action_probs = tf.nn.softmax(logits)
#             new_log_probs = tf.gather(tf.math.log(action_probs), actions, axis=1, batch_dims=1)
#             ratio = tf.exp(new_log_probs - old_log_probs)
#             clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
#             actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
        
#         policy_gradients = tape.gradient(actor_loss, policy.trainable_variables)
#         policy_optimizer.apply_gradients(zip(policy_gradients, policy.trainable_variables))

#         with tf.GradientTape() as tape:
#             values = value(states)
#             critic_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(returns, values))
        
#         value_gradients = tape.gradient(critic_loss, value.trainable_variables)
#         value_optimizer.apply_gradients(zip(value_gradients, value.trainable_variables))

# def train():
#     state = env.reset()
#     state = tf.convert_to_tensor(state, dtype=tf.float32)
#     ep_rewards = []
    
#     states = []
#     actions = []
#     rewards = []
#     dones = []
#     values = []
#     log_probs = []

#     for step in range(max_steps_per_episode):
#         action, value = get_action_and_value(tf.expand_dims(state, 0))
#         next_state, reward, done, _ = env.step(action.numpy())
#         next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)

#         states.append(state)
#         actions.append(action)
#         rewards.append(reward)
#         dones.append(done)
#         values.append(value)

#         state = next_state
#         ep_rewards.append(reward)

#         if done:
#             break

#     next_value = value(tf.expand_dims(next_state, 0))
#     advantages, returns = compute_advantages(rewards, values, next_value, dones)
#     states = tf.convert_to_tensor(states, dtype=tf.float32)
#     actions = tf.convert_to_tensor(actions, dtype=tf.int32)
#     advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
#     returns = tf.convert_to_tensor(returns, dtype=tf.float32)

#     logits = policy(states)
#     action_probs = tf.nn.softmax(logits)
#     old_log_probs = tf.gather(tf.math.log(action_probs), actions, axis=1, batch_dims=1)

#     ppo_update(states, actions, advantages, returns, old_log_probs)

#     return np.sum(ep_rewards)