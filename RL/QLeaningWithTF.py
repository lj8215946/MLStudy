import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the environment
env = gym.make('FrozenLake-v0')

tf.reset_default_graph()

# These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[1, 16], dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

# Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
# create lists to contain total rewards and steps per episode
jList = []
rList = []

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        done = False
        # The Q-Network
        j = 0
        if i == num_episodes - 1:
            env.render()
        for j in range(99):
            # Choose an action by greedily (with e chance of random action) from the Q-network
            # 这里的np.identity(16)[s:s+1]的这个是用过单位矩阵生成了one hot编码
            a, allQ = sess.run([predict, Qout], feed_dict={inputs1: np.identity(16)[s:s+1]})

            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            # Get new state and reward from environment
            new_state, reward, done, _ = env.step(a[0])
            if i == num_episodes - 1:
                env.render()
            # Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[new_state:new_state + 1]})
            # Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = reward + y * maxQ1
            # Train our network using target and predicted Q values
            _, W1 = sess.run([updateModel, W], feed_dict={inputs1: np.identity(16)[s:s+1], nextQ: targetQ})
            rAll += reward
            s = new_state
            if done:
                # Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
        jList.append(j)
        rList.append(rAll)

print("Percent of succesful episodes: ", str(sum(rList)/num_episodes), "%")

plt.plot(rList, label='rList')
plt.show()

plt.plot(jList, label='jList')
plt.show()
print("Cost time for every jList: ", jList)
