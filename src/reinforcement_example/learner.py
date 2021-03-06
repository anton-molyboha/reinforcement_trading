import numpy as np
import keras
from reinforcement_trading.tools import logodds_to_probs, weights_to_inds


def make_model(inputs, outputs):
    input_layer = keras.layers.Input(shape=(inputs,))
    layer1 = keras.layers.Dense(inputs + outputs, activation='tanh')
    layer2 = keras.layers.Dense(outputs, activation='tanh')
    scale_layer = keras.layers.Dense(outputs, activation='linear')
    model = keras.Model(inputs=input_layer, outputs=scale_layer(layer2(layer1(input_layer))))
    #model.compile('sgd', loss='mean_squared_error')
    model.compile('RMSProp', loss='mean_squared_error')
    return model


def make_model_elu(inputs, outputs):
    input_layer = keras.layers.Input(shape=(inputs,))
    layer1 = keras.layers.Dense((inputs + outputs)**2, activation='elu')
    layer2 = keras.layers.Dense(outputs, activation='elu')
    scale_layer = keras.layers.Dense(outputs, activation='linear')
    model = keras.Model(inputs=input_layer, outputs=scale_layer(layer2(layer1(input_layer))))
    #model.compile('sgd', loss='mean_squared_error')
    model.compile('RMSProp', loss='mean_squared_error')
    return model


class KernelModel(object):
    def __init__(self, xscale):
        self.xscale = xscale
        self.reset()

    def reset(self):
        self.X = np.empty((0, len(self.xscale)))
        self.Y = np.empty(0)

    def fit(self, X, Y, **kwargs):
        self.X = X / self.xscale[np.newaxis, :]
        self.Y = Y.copy()

    def predict(self, X, **kwargs):
        if len(self.X) == 0:
            return np.zeros((len(X), 1))
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
        else:
            batch_size = 32
        X = X / self.xscale[np.newaxis, :]

        def predict_batch(X):
            dists = (np.sum(X ** 2, axis=1)[:, np.newaxis] +
                     np.sum(self.X ** 2, axis=1)[np.newaxis, :] -
                     2 * np.dot(X, self.X.T))
            weights = np.exp(-(dists - np.min(dists, axis=1)[:, np.newaxis]) / 2)
            return (np.dot(weights, self.Y) / np.sum(weights, axis=1))

        res = np.empty(len(X))
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            res[start:end] = predict_batch(X[start:end])
        return res[:, np.newaxis]


class ScaledModel(object):
    def __init__(self, model, xscale=None, yscale=None):
        self.model = model
        self.xscale = xscale
        self.yscale = yscale

    def fit(self, X, Y, **kwargs):
        rX = X / self.xscale[np.newaxis, :] if self.xscale is not None else X
        rY = Y / self.yscale if self.yscale is not None else Y
        self.model.fit(rX, rY, **kwargs)

    def predict(self, X, **kwargs):
        rX = X / self.xscale[np.newaxis, :] if self.xscale is not None else X
        res = self.model.predict(rX, **kwargs)
        return res * self.yscale if self.yscale is not None else res


class ValueToPolicy(object):
    def __init__(self, value_model, scale=1.0):
        self.value_model = value_model
        self.num_actions = len(value_model)
        self.scale = scale

    def predict(self, states):
        res = np.array([model.predict(states)[:, 0] for model in self.value_model]) / self.scale
        # res has axes: action, batch
        # Result must have axes: batch, action
        return res.T

    def set_scale(self, scale):
        self.scale = scale


class AbstractLearner(object):
    def learn(self, state, action, reward, next_state, value_proxy):
        pass

    def learn_last(self, state, action, reward):
        pass

    def move(self, state):
        raise NotImplementedError("AbstractLearner is an abstract class")


class SimpleLearner(AbstractLearner):
    def __init__(self, state_scale, reward_scale, num_actions):
        self.move_score_model = make_model(len(state_scale) + num_actions, 1)
        self.state_scale = state_scale
        self.reward_scale = reward_scale
        self.action_vec = np.eye(num_actions)
        self.last_move_learn_boost = 10
        self.self_trust = 0.0
        self.trust_rate = 1e-5

    def _move_and_score(self, state):
        best = None
        best_score = None
        for act, actvec in enumerate(self.action_vec):
            score = self.move_score_model.predict(np.concatenate([state / self.state_scale, actvec])[np.newaxis, :])[0, 0]
            if not np.isfinite(score):
                raise AssertionError("predict({}, {}) = {}".format(state, actvec, score))
            if best_score is None or score > best_score:
                best_score = score
                best = act
        return best, best_score

    def learn(self, state, action, reward, next_state, value_proxy):
        #print("learn({}, {}, {}, {}, {})".format(state, action, reward, next_state, value_proxy))
        if not np.all(np.isfinite(state)) or not np.isfinite(reward) or not np.all(np.isfinite(next_state)):
            raise ValueError("NaNs in input: {}, {}, {}, {}".format(state, action, reward, next_state))
        next_move, next_score = self._move_and_score(next_state)
        from_next = reward / self.reward_scale + next_score
        prediction = self.move_score_model.predict(np.concatenate([state / self.state_scale, self.action_vec[action]])[np.newaxis, :])[0, 0]
        to_learn = self.self_trust * from_next + (1 - self.self_trust) * (reward + value_proxy) / self.reward_scale
        #print("old prediction: {}, score from next: {}, reward to learn: {}".\
        #      format(prediction, from_next, to_learn))
        self.move_score_model.train_on_batch(np.concatenate([state / self.state_scale, self.action_vec[action]])[np.newaxis, :],
                                             np.array([[to_learn]]))
        self.self_trust += self.trust_rate * (1 - self.self_trust)

    def learn_last(self, state, action, reward):
        for i in range(self.last_move_learn_boost):
            self.move_score_model.train_on_batch(np.concatenate([state / self.state_scale, self.action_vec[action]])[np.newaxis, :],
                                                 np.array([[reward / self.reward_scale]]))

    def move(self, state):
        move, score = self._move_and_score(state)
        return move


class MCLearner(AbstractLearner):
    def __init__(self, state_dim, num_actions, reward_scale=1.0, model_factory=make_model_elu):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.reward_scale = reward_scale
        self.value_model = [ScaledModel(model_factory(state_dim, 1), yscale=reward_scale) for i in range(num_actions)]
        #self.policy = make_model(state_dim, num_actions)
        self.policy = ValueToPolicy(self.value_model, scale=self.reward_scale)
        self.histories = []
        self.history = []
        # Helper arrays for faster training
        self.states = np.empty((0, self.state_dim))
        self.actions = np.empty(0, dtype=np.int)
        self.cumrewards = np.empty(0)
        self.cumlogprobs = np.empty(0)
        self.ranges = np.empty((0, 2), dtype=np.int)
        self.actinds = [np.empty(0, dtype=np.int) for i in range(self.num_actions)]

    def move(self, state):
        probs = logodds_to_probs(self.policy.predict(state[np.newaxis, :])[0])
        choice = np.searchsorted(np.cumsum(probs), np.random.rand())
        self.last_action_prob = probs[choice]
        return choice

    def learn(self, state, action, reward, next_state, value_proxy):
        self.history.append((np.array(state), action, reward, self.last_action_prob))

    def learn_last(self, state, action, reward):
        self.history.append((np.array(state), action, reward, self.last_action_prob))
        self.histories.append(self.history)
        self.history = []
        self._update_value_model()

    def _update_value_model(self):
        # Update data structures
        last_states, last_actions, last_rewards, last_actionprob = zip(*self.histories[-1][::-1])
        dsize = len(self.states)
        self.ranges = np.concatenate([self.ranges, np.array([[dsize, dsize + len(last_states)]])])
        dsize += len(last_states)
        self.states = np.concatenate([self.states, np.array(last_states)], axis=0)
        self.actions = np.concatenate([self.actions, np.array(last_actions, dtype=np.int)])
        self.cumrewards = np.concatenate([self.cumrewards, np.cumsum(last_rewards)])
        self.cumlogprobs = np.concatenate([self.cumlogprobs, [0], np.cumsum(np.log(last_actionprob))[:-1]])
        self.actinds = [np.arange(len(self.actions))[self.actions == i] for i in range(self.num_actions)]
        # Training
        policy_logprobs = np.log(logodds_to_probs(self.policy.predict(self.states))[np.arange(dsize), self.actions])
        policy_cumlogprobs = np.zeros(dsize)
        for start, end in self.ranges:
            policy_cumlogprobs[start + 1:end] = np.cumsum(policy_logprobs[start:end - 1])
        predicted_value = np.zeros(dsize)
        for action in range(self.num_actions):
            if len(self.actinds[action]) > 0:
                predicted_value[self.actinds[action]] = \
                    self.value_model[action].predict(self.states[self.actinds[action], :]).reshape(-1)
        weights = np.exp(policy_cumlogprobs - self.cumlogprobs)
        assert np.all(weights >= 0)
        for action in range(self.num_actions):
            inds = self.actinds[action][weights_to_inds(weights[self.actinds[action]])]
            if len(inds) > 0:
                self.value_model[action].fit(self.states[inds, :], self.cumrewards[inds])
        # Policy scale
        mse = np.dot(weights, (predicted_value - self.cumrewards) ** 2) / np.sum(weights)
        self.policy.set_scale(self.reward_scale / len(self.histories) + np.sqrt(mse))
        return


class TDLearner(AbstractLearner):
    def __init__(self, state_dim, num_actions, reward_scale=1.0, model_factory=make_model_elu):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.reward_scale = reward_scale
        self.value_model = [ScaledModel(model_factory(state_dim, 1), yscale=reward_scale) for i in range(num_actions)]
        #self.policy = make_model(state_dim, num_actions)
        self.policy = ValueToPolicy(self.value_model, scale=self.reward_scale)
        self.histories = []
        self.history = []
        # Helper arrays for faster training
        self.states = np.empty((0, self.state_dim))
        self.actions = np.empty(0, dtype=np.int)
        self.rewards = np.empty(0)
        self.logprobs = np.empty(0)
        self.ranges = np.empty((0, 2), dtype=np.int)
        self.actinds = [np.empty(0, dtype=np.int) for i in range(self.num_actions)]

    def move(self, state):
        probs = logodds_to_probs(self.policy.predict(state[np.newaxis, :])[0])
        choice = np.searchsorted(np.cumsum(probs), np.random.rand())
        self.last_action_prob = probs[choice]
        return choice

    def _state_value(self, state):
        action_values = np.array([model.predict(state[np.newaxis, :])[0, 0] for model in self.value_model])
        probs = logodds_to_probs(self.policy.predict(state[np.newaxis, :])[0, :])
        return np.dot(probs, action_values)

    def learn(self, state, action, reward, next_state, value_proxy):
        winds = [0] #weights_to_inds(np.array([self.last_action_prob]))
        #         self.value_model[action].fit(state[np.newaxis, :][winds, :],
        #                                      np.array([[reward + self._state_value(next_state)]])[winds],
        #                                      verbose=False)
        self.history.append((np.array(state), action, reward, self.last_action_prob))

    def learn_last(self, state, action, reward):
        winds = [0] #weights_to_inds(np.array([self.last_action_prob]))
        #         self.value_model[action].fit(state[np.newaxis, :][winds, :],
        #                                      np.array([[reward]])[winds],
        #                                      verbose=False)
        self.history.append((np.array(state), action, reward, self.last_action_prob))
        self.histories.append(self.history)
        self.history = []
        self._update_value_model()

    def _update_value_model(self):
        # Update data structures
        dsize = len(self.states)
        for i in range(len(self.ranges), len(self.histories)):
            last_states, last_actions, last_rewards, last_actionprob = zip(*self.histories[i][::-1])
            self.ranges = np.concatenate([self.ranges, np.array([[dsize, dsize + len(last_states)]])])
            dsize += len(last_states)
            self.states = np.concatenate([self.states, np.array(last_states)], axis=0)
            self.actions = np.concatenate([self.actions, np.array(last_actions, dtype=np.int)])
            self.rewards = np.concatenate([self.rewards, last_rewards])
            #self.logprobs = np.concatenate([self.logprobs, [0], np.log(last_actionprob)[:-1]])
            #self.actinds = [np.arange(len(self.actions))[self.actions == i] for i in range(self.num_actions)]
            self.actinds = [np.concatenate([self.actinds[i],
                                            np.arange(dsize - len(last_states), dsize)
                                            [self.actions[dsize - len(last_states):dsize] == i]])
                            for i in range(self.num_actions)]
        # Training
        policy_probs = logodds_to_probs(self.policy.predict(self.states))
        #policy_logprobs = np.log(policy_probs[np.arange(dsize), self.actions])
        #policy_cumlogprobs = np.zeros(dsize)
        #for start, end in self.ranges:
        #    policy_cumlogprobs[start + 1:end] = np.cumsum(policy_logprobs[start:end - 1])
        predicted_state_action_value = np.array([model.predict(self.states)[:, 0] for model in self.value_model]).T
        predicted_state_value = np.sum(policy_probs * predicted_state_action_value, axis=1)
        response = np.empty(self.rewards.shape)
        response[:] = self.rewards
        for start, end in self.ranges:
            response[start + 1:end] += predicted_state_value[start:end - 1]
        predicted_value = predicted_state_action_value[np.arange(dsize), self.actions]
        # weights = np.exp(policy_logprobs - self.logprobs)
        for action in range(self.num_actions):
            inds = self.actinds[action]  #[weights_to_inds(weights[self.actinds[action]])]
            if len(inds) > 0:
                self.value_model[action].fit(self.states[inds, :], response[inds],
                                             epochs=10#,
                                             #callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=2)]
                                             )
        # Policy scale
        #mse = np.dot(weights, (predicted_value - response) ** 2) / np.sum(weights)
        #self.policy.set_scale(self.reward_scale / len(self.histories) + np.sqrt(mse))
        self.policy.set_scale(self.reward_scale / np.sqrt(len(self.histories)))
        return


# Play the game
def train_play(game, learner):
    history = []
    total_reward = 0.0
    while True:
        state = game.get_state()
        #print(state)
        action = learner.move(state)
        #print(action)
        reward = game.time_step(action)
        history.append((state, action, reward))
        total_reward += reward
        if not game.alive:
            #print("DEAD! Survived for {}".format(total_reward))
            learner.learn_last(state, action, reward)
            return history, total_reward
        learner.learn(state, action, reward, game.get_state(), game.get_state()[3])

