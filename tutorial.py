import tensorflow as tf
import numpy as np

num_action = 4
num_hidden_unit = 128


def create_model(num_output, num_hidden):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_hidden_unit, activation='relu'),
        tf.keras.layers.Dense(num_action)
    ])

    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


train_model = create_model(num_action, num_hidden_unit)
target_model = create_model(num_action, num_hidden_unit)
target_model.set_weights(train_model.get_weights())


class player:
    def __init__(self, posx: int, posy: int):
        self.pos = np.array([posx, posy])

    def choose_action(self):
        action = np.argmax(np.random.random((4, 1)))
        if 0 == action:
            self.pos[0] -= 1
        elif 1 == action:
            self.pos[0] += 1
        elif 2 == action:
            self.pos[1] -= 1
        else:
            self.pos[1] += 1

    def get_pos(self) -> (int, int):
        return self.pos


class environment:
    def __init__(self, x: int, y: int, foodx: int, foody: int):
        self.size = np.array([x, y])
        self.foodpos = np.array([foodx, foody])

    def drawGame(self, playerPos):
        pass

    def getReward(self, playerPos) -> (int, bool):
        reward = -1
        isGameOver = False
        if (playerPos == self.foodpos).all():
            reward = 100
            isGameOver = True
        elif ((playerPos[0] > self.size[0])
                or (playerPos[0] < 0)
                or (playerPos[1] > self.size[1])
                or (playerPos[1] < 0)):
            reward = -100
            isGameOver = True
        return reward, isGameOver


max_step_in_episode = 1000
playerOne = player(50, 50)
environmentOne = environment(200, 200, 150, 150)


def game():
    rewards = 0
    isGameOver = False
    for _ in range(max_step_in_episode):
        playerOne.choose_action()
        prob_action = train_model(tf.expand_dims(playerOne.get_pos(), 0))
        action = tf.nn.softmax(prob_action)
        print(f"player position: {playerOne.get_pos()}, action: {action}")
        reward, isGameOver = environmentOne.getReward(playerOne.get_pos())
        rewards += reward
        if isGameOver:
            break
    print(f"Score: {rewards}")


if __name__ == "__main__":
    game()
