import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, Huber

class ActorCritic():
    """A class for training a discrete actor critic network."""
    def __init__(self, actor, critic, learning_rate, discount_factor):
        self.gamma = discount_factor
        self.actor = actor
        self.critic = critic
        self.optimizer = Adam(learning_rate, amsgrad=True)
        self.optimizer.build(actor.trainable_variables + critic.trainable_variables)
        self.critic_loss = Huber()

    # @tf.function()  
    # def act(self, observation):
    #     prediction = self.actor(observation[None, :])
    #     action = tf.random.categorical(tf.math.log(prediction), num_samples=1)
    #     return tf.squeeze(action, axis=-1)

    @tf.function()  
    def act(self, observation):
        prediction = self.actor(observation[None, :])
        action = tf.random.categorical(tf.math.log(prediction), num_samples=1)
        return tf.squeeze(action)

    @tf.function() 
    def returns(self, rewards):
        size = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=size)
        discounted_sum = tf.zeros([], dtype=tf.float32)
        for i in tf.range(size - 1, -1, -1):
            discounted_sum = rewards[i] + tf.cast(self.gamma * discounted_sum, dtype=tf.float32)
            #discounted_sum.set_shape([])
            returns = returns.write(i, discounted_sum)
        
        returns = returns.stack()
        return (returns - tf.reduce_mean(returns)) / tf.math.reduce_std(returns)
    
    @tf.function(reduce_retracing=True)
    def learn(self, states, actions, rewards, new_states, dones):
        returns = self.returns(rewards)
        actions_one_hot = tf.one_hot(actions, self.actor.output_shape[1]) 
        
        # self.actor.fit(states, actions_one_hot, sample_weight=advantages, verbose=False, batch_size=len(self.memory))
        # self.critic.fit(states, returns, verbose=False, batch_size=len(self.memory)) 

        with tf.GradientTape() as critic_tape:
            values = self.critic(states)
            loss = self.critic_loss(returns, values)
            
        critic_gradients = critic_tape.gradient(loss, self.critic.trainable_variables)

        with tf.GradientTape() as actor_tape:
            probabilities = self.actor(states)
            log_probabilities = tf.math.log(tf.reduce_sum(probabilities * actions_one_hot, axis=1))
            advantages = returns - values
            loss = -tf.reduce_mean(log_probabilities * advantages)

        actor_gradients = actor_tape.gradient(loss, self.actor.trainable_variables)
        
        self.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))  

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