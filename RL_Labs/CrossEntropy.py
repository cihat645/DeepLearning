import gym
import numpy as np, pandas as pd

env = gym.make("Taxi-v2")
env.reset()
env.render()

n_states = env.observation_space.n
n_actions = env.action_space.n

# print("n_states=%i, n_actions=%i"%(n_states, n_actions))
policy = np.ones((n_states, n_actions)) / n_actions

assert type(policy) in (np.ndarray,np.matrix)
assert np.allclose(policy,1./n_actions)
assert np.allclose(np.sum(policy,axis=1), 1)


def generate_session(policy, t_max=10 ** 4):
    """
    Play game until end or for t_max ticks.
    :param policy: an array of shape [n_states,n_actions] with action probabilities
    :returns: list of states, list of actions and sum of rewards
    """
    states, actions = [], []
    total_reward = 0.

    s = env.reset()

    for t in range(t_max):

        a = np.random.choice(range(policy.shape[1]), p = policy[s]) # p parameter takes into account the probability distribution of policy when selecting an action randomly

        new_s, r, done, info = env.step(a)

        # Record state, action and add up reward to states, actions and total_reward accordingly.
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break
    return states, actions, total_reward

s,a,r = generate_session(policy)
assert type(s) == type(a) == list
assert len(s) == len(a)
assert type(r) in [float,np.float]

#let's see the initial reward distribution
import matplotlib.pyplot as plt

def test_sample_rewards():
    sample_rewards = [generate_session(policy,t_max=1000)[-1] for _ in range(200)]

    plt.hist(sample_rewards,bins=20);
    plt.vlines([np.percentile(sample_rewards, 50)], [0], [100], label="50'th percentile", color='green')
    plt.vlines([np.percentile(sample_rewards, 90)], [0], [100], label="90'th percentile", color='red')
    plt.legend()
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.show()


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50):
    """
    Select states and actions from games that have rewards >= percentile
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i][t]

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    Please return elite states and actions in their original order
    [i.e. sorted by session number and timestep within session]

    If you're confused, see examples below. Please don't assume that states are integers (they'll get different later).
    """

    reward_threshold = np.percentile(rewards_batch, percentile)

    elite_indices = np.where(rewards_batch >= reward_threshold)[0]
    print('elite_indices = ', elite_indices)
    elite_states = [item for i in elite_indices for item in states_batch[i]]
    elite_actions = [item for i in elite_indices for item in actions_batch[i]]

    #     print('elite_states', elite_states)
    #     print('elite_actions', elite_actions)
    return elite_states, elite_actions

def test_select_elites():
    states_batch = [
        [1, 2, 3],  # game1
        [4, 2, 0, 2],  # game2
        [3, 1]  # game3
    ]

    actions_batch = [
        [0, 2, 4],  # game1
        [3, 2, 0, 1],  # game2
        [3, 3]  # game3
    ]
    rewards_batch = [
        3,  # game1
        4,  # game2
        5,  # game3
    ]

    test_result_0 = select_elites(states_batch, actions_batch, rewards_batch, percentile=0)
    test_result_40 = select_elites(states_batch, actions_batch, rewards_batch, percentile=30)
    test_result_90 = select_elites(states_batch, actions_batch, rewards_batch, percentile=90)
    test_result_100 = select_elites(states_batch, actions_batch, rewards_batch, percentile=100)

    assert np.all(test_result_0[0] == [1, 2, 3, 4, 2, 0, 2, 3, 1]) \
           and np.all(test_result_0[1] == [0, 2, 4, 3, 2, 0, 1, 3, 3]), \
        "For percentile 0 you should return all states and actions in chronological order"
    assert np.all(test_result_40[0] == [4, 2, 0, 2, 3, 1]) and \
           np.all(test_result_40[1] == [3, 2, 0, 1, 3, 3]), \
        "For percentile 30 you should only select states/actions from two first"
    assert np.all(test_result_90[0] == [3, 1]) and \
           np.all(test_result_90[1] == [3, 3]), \
        "For percentile 90 you should only select states/actions from one game"
    assert np.all(test_result_100[0] == [3, 1]) and \
           np.all(test_result_100[1] == [3, 3]), \
        "Please make sure you use >=, not >. Also double-check how you compute percentile."
    print("Ok!")
    return True


def update_policy(elite_states, elite_actions):
    """
    Given old policy and a list of elite states/actions from select_elites,
    return new updated policy where each action probability is proportional to

    policy[s_i,a_i] ~ #[occurences of si and ai in elite states/actions]

    Don't forget to normalize policy to get valid probabilities and handle 0/0 case.
    In case you never visited a state, set probabilities for all actions to 1./n_actions

    :param elite_states: 1D list of states from elite sessions
    :param elite_actions: 1D list of actions from elite sessions

    """

    new_policy = np.zeros([n_states, n_actions])

    #   update probabilities for actions given elite states & actions
    for s in range(n_states):
        num_visits = np.sum(np.array(elite_states) == s)  # num_visits to s in elite states
        indices_of_s = np.where(np.array(elite_states) == s)[
            0]  # indices of state s (used to only consider actions from state s)
        actions_of_s = [elite_actions[i] for i in indices_of_s]  # list of all actions taken from state s
        if num_visits > 0:
            new_policy[s] = [np.sum(np.array(actions_of_s) == action) / num_visits for action in range(n_actions)]
        else:
            new_policy[s] = 1 / n_actions  # set 1/n_actions for all actions in unvisited states.

    return new_policy

def test_update_policy():
    elite_states, elite_actions = ([1, 2, 3, 4, 2, 0, 2, 3, 1], [0, 2, 4, 3, 2, 0, 1, 3, 3])

    new_policy = update_policy(elite_states, elite_actions)
    print(new_policy)
    assert np.isfinite(new_policy).all(), "Your new policy contains NaNs or +-inf. Make sure you don't divide by zero."
    assert np.all(new_policy >= 0), "Your new policy can't have negative action probabilities"
    assert np.allclose(new_policy.sum(axis=-1),
                       1), "Your new policy should be a valid probability distribution over actions"
    reference_answer = np.array([
        [1., 0., 0., 0., 0.],
        [0.5, 0., 0., 0.5, 0.],
        [0., 0.33333333, 0.66666667, 0., 0.],
        [0., 0., 0., 0.5, 0.5]])
    assert np.allclose(new_policy[:4, :5], reference_answer)
    print("Ok!")


from IPython.display import clear_output
def show_progress(batch_rewards, log, percentile, reward_range=[-990, +10]):
    """
    A convenience function that displays training progress.
    No cool math here, just charts.
    """

    # clear_output(True)
    print("mean reward = %.3f, threshold=%.3f" % (log[-1][0], log[-1][1]))
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

#reset policy just in case
policy = np.ones([n_states, n_actions]) / n_actions

n_sessions = 400  # sample this many sessions
percentile = 50  # take this percent of session with highest rewards
learning_rate = 0.5  # add this thing to all counts for stability

log = []

for i in range(100):
    sessions = [generate_session(policy, 1000) for _ in range(n_sessions)]  # <generate a list of n_sessions new sessions>]

    batch_states, batch_actions, batch_rewards = zip(*sessions)

    elite_states, elite_actions = select_elites(batch_states, batch_actions, batch_rewards, percentile)

    new_policy = update_policy(elite_states, elite_actions)

    policy = learning_rate * new_policy + (1 - learning_rate) * policy
    log.append([np.mean(batch_rewards), np.percentile(batch_rewards, percentile)])  # mean rewards, elite threshold
    # display results on chart
show_progress(batch_rewards, log, percentile)

# Find out how the algorithm performance changes if you change different percentile and different n_samples
# -- Let's see what happens if we make the elite more selective (i.e. only update policy with respect to the sessions which performed
# very well relative to the rest of them. Say, a percentile of 75.
#    TEST:   hyperparameter settings: n_sessions = 250, percentile = 75, learning_rate = 0.5
#    RESULTS: mean reward = -324.984, threshold=7.750
#   ---> performance actually decreased with more iterations!

# Let's try a percentile setting of 90%
#  TEST: hyperparameter settings: n_sessions = 250, percentile = 90, learning_rate = 0.5
#  RESULTS: Stopped early --> performance also significantly decreased and was not converging

# TEST: hyperparameter settings: n_sessions = 250, percentile = 30, learning_rate = 0.5
# RESULTS:   So far, best results! mean reward = -7.624, threshold=3.700, we even had positive mean reward for some point, but it
#   didn't converge on a positive expected value.

# TEST: hyperparameter settings: n_sessions = 400, percentile = 50, learning_rate = 0.5
# RESULTS: Okay results, but increasing n_sessions doesn't seem to affect the expected mean very much
# mean reward = -66.183, threshold=8.000

# TEST: hyperparameter settings: n_sessions = 250, percentile = 10, learning_rate = 0.5
# RESULTS: Best results! Finally achieved a positive mean reward!
#  mean reward = 2.948, threshold=-12.000
