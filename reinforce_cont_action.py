import gym
import numpy as np
import scipy.special as sci
import roboschool
import tensorflow as tf
import pdb

# instantiated for both mean and standard deviation
#
# input of network takes the state (observation)
# output of the network is a single float
class PolicyComponent():
    def __init__(self, hparams, session, learning_rate):
        with tf.name_scope('policy_component'):
            self.learning_rate = learning_rate
            self.session = session
            self.advantages = tf.placeholder(tf.float32, name="advantages")
            self.observation = tf.placeholder(tf.float32, shape=[None, hparams['observation_size']])
            self.build_network(hparams)
            self.build_trainer(hparams)

    def build_network(self, hparams):
        hidden1 = tf.contrib.layers.fully_connected(
                inputs=self.observation,
                num_outputs=hparams['hidden_size1'],
                activation_fn=tf.nn.relu,
                weights_initializer=tf.random_normal_initializer())

        single = tf.contrib.layers.fully_connected(
                inputs=hidden1,
                num_outputs=1,
                activation_fn=None)

        self.sample = single #tf.sigmoid(single)

    def build_trainer(self, hparams):
        loss = self.build_loss()
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.train = optimizer.minimize(loss)

    def build_loss(self):
        reduced_sum = tf.reduce_sum(tf.multiply(self.sample, self.advantages))
        return -reduced_sum

    # given an observation, output the policy component
    def observe_and_act(self, observation):
        return self.session.run(self.sample, feed_dict={self.observation: [observation]})

    def train_net(self, batch_observations, batch_advantages):
        feed = {
            self.observation: batch_observations,
            self.advantages: batch_advantages
        }
        return self.session.run(self.train, feed_dict = feed)

def policy_rollout(env, mean_component, std_component):
    """Run one episode."""

    observation, reward, done = env.reset(), 0, False
    observations, actions, rewards  = [], [], []

    while not done:
        # still_open = env.render("human")
        # if still_open == False:
        #     return
        observations.append(observation)
        orig_mean = mean_component.observe_and_act(observation)
        mean = orig_mean # (orig_mean - 0.5) * 4 # shifting sigmoid values to between (-2,2)
        orig_std = std_component.observe_and_act(observation)
        std = np.exp(orig_std) # e^x suggestion by Sutton Chapter 13.7
        # pdb.set_trace()
        action = np.random.normal(mean[0], std[0])
        # print("{}:{}".format(orig_mean[0][0], orig_std[0][0]), end=", ")
        observation, reward, done, _ = env.step(action)

        actions.append(action)
        rewards.append(reward)

    return observations, actions, rewards

def process_rewards(rewards):
    """Rewards -> Advantages for one episode. """

    # total reward: length of episode
    return [len(rewards)] * len(rewards)

def main():
    env = gym.make('RoboschoolInvertedPendulum-v2') # Raw Theta

    # hyper parameters
    hparams = {
            'observation_size': env.observation_space.shape[0], # should be 4
            'hidden_size1': 36,
            'mean_learning_rate': 0.001,
            'std_learning_rate': 0.01
    }

    # environment params
    eparams = {
            'num_batches': 100,
            'ep_per_batch': 10
    }

    with tf.Graph().as_default(), tf.Session() as sess:
        mean_component = PolicyComponent(hparams, sess, hparams['mean_learning_rate'])
        std_component = PolicyComponent(hparams, sess, hparams['std_learning_rate'])

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./logs/", graph=tf.get_default_graph())

        for batch in range(eparams['num_batches']):
            print('\n=====\nBATCH {}\n===='.format(batch))
            batch_observations, batch_actions, batch_rewards, observation_lengths = [], [], [], []

            for ep_index in range(eparams['ep_per_batch']):
                observations, actions, rewards = policy_rollout(env, mean_component, std_component)
                observation_length = len(observations)
                observation_lengths.extend([observation_length])
                print('Episode {} steps: {}'.format((ep_index+1)+(10 * batch),observation_length))
                batch_observations.extend(observations)
                batch_actions.extend(actions)
                advantages = process_rewards(rewards)
                batch_rewards.extend(advantages)

            print(">> Avg Steps: {}".format(np.average(observation_lengths)))
            # normalize rewards; don't divide by 0
            batch_rewards = (batch_rewards - np.mean(batch_rewards)) / (np.std(batch_rewards) + 1e-10)
            mean_component.train_net(batch_observations, batch_rewards)
            std_component.train_net(batch_observations, batch_rewards)

if __name__ == "__main__":
    main()
