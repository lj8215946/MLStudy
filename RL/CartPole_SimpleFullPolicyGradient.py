import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym

# 加载平衡小车的虚拟环境
env = gym.make('CartPole-v0')

gamma = 0.99


def discount_rewards(rewards):
    """ 
    take 1D float array of rewards and compute discounted reward
    
    Parameters
    ----------
    rewards : 立即获得的奖励数组，数组的尾部(rewards[rewards.size - 1])是最早获得的奖励，头部(rewards[0])是最新获得的奖励
    """
    discounted_r = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add

    return discounted_r


class Agent:
    def __init__(self, lr, s_size, a_size, h_size):
        """
        创建一个描述了平衡车项目的TF计算过程
        
        Parameters
        ----------
        :param lr:  学习速率
        :param s_size: 状态空间纬度
        :param a_size: 可能的动作个数
        :param h_size: hidden layout的神经元个数
        """

        # These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        # The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        # to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        # tf.shape(self.output)[0]是批处理的size，tf.shape(self.output)[1]是action的size
        # tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1]选择了每此选择的第一项
        # self.indexes打成1D之后每个选择的项
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        # 将所有被选中的输出项集中在一起
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        # 使用policy gradient策略计算loss function
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        trainable_variables = tf.trainable_variables()
        self.gradient_holders = []
        for each_idx, var in enumerate(trainable_variables):
            placeholder = tf.placeholder(tf.float32, name=str(each_idx) + '_holder')
            self.gradient_holders.append(placeholder)

        # 该方法计算了loss对每一个trainable_variables的偏导（其实就是梯度），返回一个len（trainable_variables）的数组，每一项为loss对每一个trainable_variable求偏导的总和，
        self.gradients = tf.gradients(self.loss, trainable_variables)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # 这一步根据gradient_holders的梯度值更新trainable_variables
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, trainable_variables))


# 开始训练
tf.reset_default_graph()  # Clear the Tensorflow graph.

myAgent = Agent(lr=1e-2, s_size=4, a_size=2, h_size=8)  # Load the agent.

# 做5000次这个游戏
total_episodes = 5000  # Set total number of episodes to train agent on.
# 没做一次游戏最大的步数
max_ep = 999
update_frequency = 5

init = tf.global_variables_initializer()

# Launch the tensorflow graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_length = []

    # 下面几行创建了一个list，对所有的trainable_variables创建一个shape相同的全零的数组
    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    # 循环total_episodes多次游戏
    while i < total_episodes:
        s = env.reset()
        running_reward = 0# 记录总奖励值
        # 存储每一步选择获的【老状态，动作，奖励，新状态】
        ep_history = []

        # 最大步数循环
        for j in range(max_ep):
            # Probabilistically pick an action given our network outputs.
            # 按照action的概率选择一个action
            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in: [s]})
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)

            # 按照给定的Action进行动作的选择
            s1, r, d, _ = env.step(a)  # Get our reward for taking an action given a bandit.
            # 每一次reward都是1这个为什么会有效果呢？
            # if d:
            #     r = -10
            ep_history.append([s, a, r, s1])
            s = s1
            running_reward += r
            if d:
                # Update the network.
                ep_history = np.array(ep_history)
                # 这里是非常关键的一步，改一步进行了所谓的对未来的收益打折的处理
                ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                # state是一个len==4的数组，numpy的二维数组和数组嵌套数组不是同一个东西，需要用vstack来将数组套数组转化为二维数组。
                feed_dict = {myAgent.reward_holder: ep_history[:, 2],
                             myAgent.action_holder: ep_history[:, 1], myAgent.state_in: np.vstack(ep_history[:, 0])}
                grads = sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                    # 这里大费周章的将梯度拿出来专门搞其实就是为了没update_frequency（5）次，进行一次权值的更新相当于一个小的batch
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad * 0

                total_reward.append(running_reward)
                total_length.append(j)
                break

                # Update our running tally of scores.
        if i % 100 == 0:
            print("Total Reward:", np.mean(total_reward[-100:]), "，Total Step:", np.mean(total_length))
        i += 1
