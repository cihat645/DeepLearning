import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from IPython.display import clear_output
import time
from joblib import Parallel, delayed

class Agent(object):

    def __init__(self, a_model):
        self.model = a_model      # rn, assuming its sklearn nn

    def generate_session(self, env_name, t_max=1000):
        """
        Generates several sessions of following current policy in the environment.
        Input: env_name - string  - used to initalize an environment.
                t_max - int - maximum number of timesteps"""

        states, actions = [], []
        total_reward = 0
        gym.logger.set_level(40)  # Doesn't display the first 40 Warning messages to console
        env = gym.make(env_name)    # create environment
        s = env.reset()             # set initial state

        for t in range(t_max):

            probs = self.model.predict_proba([s])[0]  # a vector of action probabilities in current state
            a = np.random.choice(range(len(probs)), p=probs)  # sample using probs as probability distribution
            new_s, r, done, info = env.step(a)

            # record sessions
            states.append(s)
            actions.append(a)
            total_reward += r
            s = new_s
            if done: break
        return states, actions, total_reward # each session is a list comprised of a list of states, a list of actions and the total reward. [[list_of_states],[list_of_actions],reward]

    def select_elites(self, states_batch, actions_batch, rewards_batch, percentile=50):
        """
        Select states and actions from games that have rewards >= percentile
        :param states_batch: list of lists of states, states_batch[session_i][t]
        :param actions_batch: list of lists of actions, actions_batch[session_i][t]
        :param rewards_batch: list of rewards, rewards_batch[session_i][t]

        :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

        Return elite states and actions in their original order
        [i.e. sorted by session number and timestep within session]
        """
        reward_threshold = np.percentile(rewards_batch, percentile)
        elite_indices = np.where(rewards_batch > reward_threshold)[0]
        elite_states = [s for i in elite_indices for s in states_batch[i]]
        elite_actions = [a for i in elite_indices for a in actions_batch[i]]

        return elite_states, elite_actions

    def plot_training(self, batch_rewards, log, percentile, reward_range=[-990, +10]):
        """
        A function that displays training progress.
        """
        clear_output(True)
        # print("mean reward = %.3f, threshold=%.3f" % (mean_reward, threshold))
        plt.figure(figsize=[8, 4])
        plt.subplot(1, 2, 1)
        plt.plot(list(zip(*log))[0], label='Mean rewards')
        plt.plot(list(zip(*log))[1], label='Reward thresholds')
        plt.legend()
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.hist(batch_rewards, range=reward_range);
        plt.vlines([np.percentile(batch_rewards, percentile)], [0], [100], label="percentile", color='red')
        plt.legend()
        plt.grid()
        plt.show()

    # # TODO Decide if you should amalgamate the Train functions into one function that accepts an environment parameter as input
    def TrainLunarLander(self, n_sessions = 100, percentile = 50, iters = 100, parallel = None, display_training = False):    # CANNOT HAVE Parallel(n_jobs = N) as a default function parameter
        won = False
        env = gym.make("LunarLander-v2")
        env.reset()         #  creates a new environment
        num_actions = env.action_space.n
        previous_sessions, num_p_s = [], 4    # num_p_s = number of previous sessions we will cache and use in the nex iteration
        self.model.fit([env.reset()] * num_actions, list(range(num_actions)));   # want num_actions output nodes
        rewards, percentiles, log = [], [], []

        for i in range(iters):            # iterations
            print("i = " + str(i))
            # previous_sessions = parallel(delayed(self.generate_session)("LunarLander-v2") for _ in range(n_sessions))
            new_sessions = parallel(delayed(self.generate_session)("LunarLander-v2") for _ in range(n_sessions//num_p_s)) # generate n_sessions in parallel because they are independent, sessions = [([list_of_states],[list_of_actions],reward), ([states],[actions],reward), (...) ]
            previous_sessions += new_sessions

            if len(previous_sessions) == n_sessions:
                del previous_sessions[0:n_sessions//num_p_s]           # must begin deleting previous sessions after num_p_s threshold is reached

            # turn sessions into np.arrays and then zips them up into 3-tuples,
            batch_states, batch_actions, batch_rewards = map(np.array, zip(*previous_sessions)) # goal is to produce: tuple of state lists, tuple of action lists, tuple of rewards, then convert them to np.arrays

            elite_states, elite_actions = self.select_elites(batch_states, batch_actions, batch_rewards, percentile)
            self.model.fit(elite_states, elite_actions)                             #  fit model to predict elite_actions from elite_states
            mean_reward, threshold = np.mean(batch_rewards), np.percentile(batch_rewards, percentile)
            log.append([mean_reward, threshold])
            print("Mean reward = ", mean_reward)

            if np.mean(batch_rewards) > 50:
                won = True
                print("You Win Lunar Landing!")
                break

        if not won:
            print("Game over!")
        if display_training:
            self.plot_training(batch_rewards, log, percentile, reward_range=[min(map(lambda x: x[0], log)), np.max(batch_rewards)])  # want to get minimum number of a list of tuples
        env.close()
        return mean_reward


    def TrainCartPole(self, n_sessions = 100, percentile = 70):
        env = gym.make("CartPole-v0").env
        env.reset()
        n_actions = env.action_space.n

        # initialize model to the dimension of state an amount of actions
        self.model.fit([env.reset()] * n_actions, list(range(n_actions)));  # the ';' prevents the model declaration from printing
        rewards, percentiles, log = [], [], []

        for i in range(100):
            # if i % 25 == 0:
                # env = gym.wrappers.Monitor(gym.make("CartPole-v0"), directory="videos", force=True)

            sessions = [self.generate_session(env) for _ in range(n_sessions)]              # generate new sessions
            batch_states, batch_actions, batch_rewards = map(np.array, zip(*sessions))          # turn sessions into np.arrays and then zips them up into 3-tuples
            elite_states, elite_actions = self.select_elites(batch_states, batch_actions, batch_rewards, percentile)
            self.model.fit(elite_states, elite_actions)                             #  fit model to predict elite_actions from elite_states
            mean_reward, threshold = np.mean(batch_rewards), np.percentile(batch_rewards, percentile)
            log.append([mean_reward, threshold])

            if np.mean(batch_rewards) > 190:
                print("You Win! You may stop training now via KeyboardInterrupt.")
                break

        self.plot_training(batch_rewards, log, percentile, reward_range=[0, np.max(batch_rewards)])

        # Recording gym sessions: https://github.com/openai/gym-recording
        # env = gym.wrappers.Monitor(gym.make("CartPole-v0"), directory="videos", force=True)
        # sessions = [self.generate_session() for _ in range(100)]
        env.close()


def test_times():    # Experiment to see what is the best number of cores for 15 iterations

    num_cores = 4; times = []  # Tuples (num_cores, time)
    for i in range(1, num_cores + 1):
        model = MLPClassifier(hidden_layer_sizes=(40,40), activation='relu', warm_start= True, max_iter = 1)
        agent = Agent(model)
        start = time.time()   # Since the weights are the same, I need to
        agent.TrainLunarLander(iters = 15, parallel = Parallel(n_jobs = i))
        times.append((i, time.time() - start))
    print(times)



agent1 = MLPClassifier(hidden_layer_sizes=(20, 20),
                              activation='tanh',
                              warm_start=True,  # keep progress between .fit(...) calls
                              max_iter=1  # make only 1 iteration on each .fit(...)
                              )


# _007 = Agent(agent1)
# _007.TrainCartPole()
# lander = Agent(agent2)
# start = time.time()
# lander.TrainLunarLander()
# print("Time elapsed:", time.time() - start)

np.random.seed(1)

if __name__ == "__main__":
    gym.logger.set_level(40)  # Doesn't display the first 40 Warning messages to console

    model1 = MLPClassifier(hidden_layer_sizes=(40, 40), activation='relu', warm_start=True, max_iter=1)
    # warm_start = When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
    # max_iter = maximum number of iterations, but with stochastic solvers ('sgd', 'adam') this denotes the number of epochs
    model2 = MLPClassifier(hidden_layer_sizes=(40, 40), activation='tanh', warm_start=True, max_iter=1)
    model4 = MLPClassifier(hidden_layer_sizes=(40,40), activation = 'tanh', warm_start=True, max_iter = 1, learning_rate_init = 0.25)

    buzz_aldrin = Agent(model2)
    start = time.time()
    reward = buzz_aldrin.TrainLunarLander(iters = 500, n_sessions= 200, parallel = Parallel(n_jobs=4), display_training= True, percentile=30)
    print("Model1 Training time:" + str(time.time() - start))
    print("End mean reward" + str(reward))



    # agent3 = MLPClassifier(hidden_layer_sizes=(40, 40), activation='relu', warm_start=True, max_iter=1)








    # Q: How can I view the weights and biases of the MLP classifier? I need to do this so I can see if I can use the same 'agent2' to initialize many agents. Do the weights of the trained agent2 = the initial weights of agent3?
    # lander = Agent(agent2)
    # lander.TrainLunarLander(iters = 3, parallel = Parallel(n_jobs=2))
    # lander2 = Agent(agent2)
    # print(lander2.model.coefs_ == lander.model.coefs_)   # print lander2's model weights,
    #
    # print("Number of iterations for lander: ", lander.model.n_iter_)


    # ----- Working -------------
    # a3 = Agent(agent3)
    # start = time.time()
    # matrices = [a3.initialize_random_matrices(1000) for _ in range(100)]
    # print("Time elapsed:" + str(time.time() - start))
    # start = time.time()
    # matrices = a3.parallel_matrices(100, Parallel(n_jobs = 2))
    # print("Time elapsed:" + str(time.time() - start))
    # ----- End Working -------------



    # ----- Working -------------
    # MATRICES = initialize_matrices2(100)   # WORKS!!!
    # simple1 = Simple()                        # GOT IT WORKING ON A CUSTOM CLASS "SIMPLE()"
    # start = time.time()
    # ans = simple1.initialize_matrices2(100, Parallel(n_jobs=2))
    # print("Time elapsed: " + str(time.time()-start))
    # ----- End Working -------------


    # EXAMPLE:
    # print([np.random.rand() for _ in range(100)])
    # [np.sqrt( i ** 2) for i in range(10)]
    # print("Normal time:", time.time() - start)
    #
    # # Parallel(n_jobs = 2,)
    # start = time.time()
    # Parallel(n_jobs=2)(delayed(np.sqrt)(i ** 2) for i in range(10))
    # print("2 cores time:" + str(time.time() - start))



