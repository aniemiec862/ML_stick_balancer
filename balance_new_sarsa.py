import math
import random
import numpy as np
import gym


class QLearner:
    def __init__(self):
        self.environment = gym.make('CartPole-v1')
        # self.environment = gym.make('CartPole-v1', render_mode="human")
        self.attempt_no = 1
        self.upper_bounds = [
            self.environment.observation_space.high[0],
            0.5,
            self.environment.observation_space.high[2],
            math.radians(50)
        ]
        self.lower_bounds = [
            self.environment.observation_space.low[0],
            -0.5,
            self.environment.observation_space.low[2],
            -math.radians(50)
        ]


        self.num_buckets = 4
        self.bucket_widths = [
            (self.upper_bounds[0] - self.lower_bounds[0]) / self.num_buckets,
            (self.upper_bounds[1] - self.lower_bounds[1]) / self.num_buckets,
            (self.upper_bounds[2] - self.lower_bounds[2]) / self.num_buckets,
            (self.upper_bounds[3] - self.lower_bounds[3]) / self.num_buckets
        ]

        self.Q = {}
        self.epsilon = 1
        self.learning_rate = 1
        self.discount_factor = 0.99

    def learn(self, max_attempts, output_file):
        with open(output_file, 'w') as file:
            for attempt_id in range(max_attempts):
                reward_sum = self.attempt()
                self.epsilon = max(0.01, np.exp(-0.001 * attempt_id))
                self.learning_rate = max(0.01, np.exp(-0.001 * attempt_id))

                # print(reward_sum)
                file.write(str(reward_sum) + '\n')

    def attempt(self):
        observation = self.discretise(self.environment.reset()[0])
        action = self.pick_action(observation)
        done = False
        reward_sum = 0.0
        while not done:
            self.environment.render()
            new_observation, reward, done, _, info = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            new_action = self.pick_action(observation)
            self.update_knowledge(action, observation, new_action, new_observation, reward)
            observation = new_observation
            action = new_action
            reward_sum += reward
        self.attempt_no += 1
        return reward_sum

    def discretise(self, observation):
        parameters = []
        for parameter_id in range(4):
            found = False
            for bucket_id in range(self.num_buckets):
                if observation[parameter_id] <= self.lower_bounds[parameter_id] + self.bucket_widths[parameter_id] * (bucket_id + 1):
                    parameters.append(bucket_id + 1)
                    found = True
                    break
            # last case, when value is above the established upper bound
            if found is False:
                parameters.append(self.num_buckets)
        return parameters[0], parameters[1], parameters[2], parameters[3]

    def pick_action(self, observation):
        if random.uniform(0, 1) < self.epsilon:
            return self.environment.action_space.sample()
        else:
            # 0 - go left, 1 - go right
            if (observation, 0) not in self.Q or (observation, 1) not in self.Q:
                # If there are no Q values for the current observation, choose based on observation[2]
                return 0 if observation[2] < self.num_buckets // 2 else 1

            return 0 if self.Q.get((observation, 0), 0) > self.Q.get((observation, 1), 1) else 1

    def update_knowledge(self, action, observation, new_action, new_observation, reward):
        # check if an action for this move already exists
        if (observation, action) not in self.Q:
            self.Q[(observation, action)] = 0.0

        current_value = self.Q[(observation, action)]
        future_value = self.Q.get((new_observation, new_action), 0)
        new_value = (
            (1 - self.learning_rate) * current_value +
            self.learning_rate * (reward + self.discount_factor * future_value)
        )
        # update the Q value for the current state and action
        self.Q[(observation, action)] = new_value


def main():
    for i in range(1):
        output_file = f"result_sarsa{i+1}.txt"
        learner = QLearner()
        learner.learn(10000, output_file)


if __name__ == '__main__':
    main()
