import numpy as np
import keras


def make_model(inputs, outputs):
    input_layer = keras.layers.Input(shape=(inputs,))
    layer1 = keras.layers.Dense(inputs + outputs, activation='tanh')
    layer2 = keras.layers.Dense(outputs, activation='tanh')
    scale_layer = keras.layers.Dense(outputs, activation='linear')
    model = keras.Model(inputs=input_layer, outputs=scale_layer(layer2(layer1(input_layer))))
    #model.compile('sgd', loss='mean_squared_error')
    model.compile('RMSProp', loss='mean_squared_error')
    return model


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

