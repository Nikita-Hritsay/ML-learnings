import gymnasium
import numpy as np

class QAlgorythm:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1, seed=42):
        self.env = gymnasium.make("CartPole-v1", render_mode="human")
        self.n_actions = self.env.action_space.n
        
        # Дискретизація станів
        self.bins = [10, 10, 10, 10]  # кількість бінів для кожного параметра стану
        self.observation_space_low = self.env.observation_space.low
        self.observation_space_high = self.env.observation_space.high
        self.observation_space_high[1] = 5  # обмежимо значення для швидкості візка
        self.observation_space_low[1] = -5
        self.observation_space_high[3] = np.radians(50)  # обмежимо значення для швидкості кута полюса
        self.observation_space_low[3] = -np.radians(50)

        self.alpha = alpha  # швидкість навчання
        self.gamma = gamma  # дисконтний фактор
        self.epsilon = epsilon  # ймовірність випадкового вибору дії
        self.seed = seed  # випадкове насіння для відтворюваності результатів

        # Q-таблиця, що зберігає значення для кожного стану та дії
        self.Q = np.zeros(self.bins + [self.n_actions])
        print(self.Q.shape)

    def discretize_state(self, observation):
        # Перетворює безперервні значення стану в дискретні індекси
        ratios = [(observation[i] - self.observation_space_low[i]) / 
                  (self.observation_space_high[i] - self.observation_space_low[i]) for i in range(len(observation))]
        new_observation = [int(round((self.bins[i] - 1) * ratios[i])) for i in range(len(observation))]
        new_observation = [min(self.bins[i] - 1, max(0, new_observation[i])) for i in range(len(new_observation))]
        return tuple(new_observation)

    def main(self):
        # Кількість епізодів навчання
        for episode in range(1000):
            # Скидання середовища для нового епізоду
            observation, info = self.env.reset(seed=self.seed)
            # Дискретизація початкового стану
            state = self.discretize_state(observation)
            # Нагорода за епізод
            total_reward = 0

            for _ in range(1000):
                if np.random.uniform(0, 1) < self.epsilon:
                    # Випадковий вибір дії (експлорація)
                    action = self.env.action_space.sample()
                else:
                    # Вибір дії з найбільшою нагородою (експлуатація)
                    action = np.argmax(self.Q[state])

                # Виконання дії та отримання нових спостережень
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                
                # Дискретизація нового стану
                next_state = self.discretize_state(next_observation)

                # Додавання нагороди за крок
                total_reward += reward

                # Вибір найкращої наступної дії для нового стану
                best_next_action = np.argmax(self.Q[next_state])

                # Оновлення значення Q для поточного стану та дії
                self.Q[state][action] = self.Q[state][action] + self.alpha * (
                    reward + self.gamma * self.Q[next_state][best_next_action] - self.Q[state][action]
                )

                # Перехід до нового стану
                state = next_state

                if terminated or truncated:
                    break

            print(f"Episode {episode} Total Reward: {total_reward}")

qAlgorythm = QAlgorythm()
qAlgorythm.main()
