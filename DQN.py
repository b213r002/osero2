import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import os
from tensorflow.keras import layers, models
import pickle  # Qテーブル保存用

def create_dqn_model(input_shape, num_actions):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))  # 入力の盤面を1Dに変換
    model.add(layers.Dense(64, activation='relu'))  # 隠れ層
    model.add(layers.Dense(64, activation='relu'))  # 隠れ層
    model.add(layers.Dense(num_actions, activation='linear'))  # 出力層: 各行動に対するQ値を出力
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def choose_action_dqn(model, state, epsilon, num_actions, is_black_turn):
    if random.random() < epsilon:
        # ランダムに選択
        valid_moves = get_valid_moves(get_board_from_state_str(state), is_black_turn)
        return random.choice(valid_moves)
    else:
        # 状態に対するQ値を予測
        q_values = model.predict(np.array([get_board_from_state_str(state)]))
        action_index = np.argmax(q_values[0])  # Q値が最大のインデックスを選択
        
        # インデックスから行動に対応する(row, col)を取得
        row = action_index // 8
        col = action_index % 8
        return (row, col)

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)  # バッファがいっぱいになったら古い経験を削除

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)  # バッチサイズ分の経験をランダムにサンプリング

    def size(self):
        return len(self.buffer)

def update_dqn_model(model, buffer, batch_size, gamma):
    if buffer.size() < batch_size:
        return  # 経験リプレイバッファが満たされていない場合は更新しない

    batch = buffer.sample(batch_size)
    states, actions, rewards, next_states, done_flags = zip(*batch)

    # 次の状態における最大Q値を予測
    # ここで状態をget_board_from_state_strでNumPy配列に変換する
    next_states_array = np.array([get_board_from_state_str(state_str) for state_str in next_states])
    next_q_values = model.predict(next_states_array)
    next_q_max = np.max(next_q_values, axis=1)

    # Q値のターゲット
    targets = rewards + (gamma * next_q_max * (1 - np.array(done_flags)))

    # 状態におけるQ値を計算
    states_array = np.array([get_board_from_state_str(state_str) for state_str in states])
    q_values = model.predict(states_array)
    for i, action in enumerate(actions):
        q_values[i][action[0] * 8 + action[1]] = targets[i]  # actionは(row, col)なので、インデックスを1次元に変換  # 選択された行動に対してターゲット値を更新

    model.fit(states_array, q_values, epochs=1, verbose=0)  # モデルを更新


# 盤面の初期化
def init_board():
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = board[4, 4] = -1  # 白
    board[3, 4] = board[4, 3] = 1   # 黒
    return board

# 石を置けるか確認する関数
def can_put(board, row, col, is_black_turn):
    if board[row, col] != 0:
        return False
    player = 1 if is_black_turn else -1
    opponent = -player
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in directions:
        x, y = row + dx, col + dy
        found_opponent = False
        while 0 <= x < 8 and 0 <= y < 8 and board[x, y] == opponent:
            found_opponent = True
            x += dx
            y += dy
        if found_opponent and 0 <= x < 8 and 0 <= y < 8 and board[x, y] == player:
            return True
    return False

# 石を置く関数
def put(board, row, col, is_black_turn):
    player = 1 if is_black_turn else -1
    board[row, col] = player
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in directions:
        x, y = row + dx, col + dy
        stones_to_flip = []
        while 0 <= x < 8 and 0 <= y < 8 and board[x, y] == -player:
            stones_to_flip.append((x, y))
            x += dx
            y += dy
        if 0 <= x < 8 and 0 <= y < 8 and board[x, y] == player:
            for fx, fy in stones_to_flip:
                board[fx, fy] = player

# 勝者判定のために石の数をカウントする関数
def count_stones(board):
    black = np.sum(board == 1)
    white = np.sum(board == -1)
    return black, white

# 盤面を表示する関数
def print_board(board):
    for row in board:
        print(" ".join(["●" if x == 1 else "○" if x == -1 else "." for x in row]))
    print()

def get_valid_moves(board, is_black_turn):
    return [(row, col) for row in range(8) for col in range(8) if can_put(board, row, col, is_black_turn)]


def get_state_str(board):
    return str(board.flatten())

    
# 状態文字列を元の盤面（NumPy配列）に戻す関数
def get_board_from_state_str(state_str):
    state_array = np.fromstring(state_str.strip("[]"), sep=" ", dtype=int)
    return state_array.reshape((8, 8))
    
def play_game_dqn(model, replay_buffer, batch_size, alpha, gamma, epsilon):
    board = init_board()
    is_black_turn = True
    total_reward = 0

    while True:
        state = get_state_str(board)
        actions = get_valid_moves(board, is_black_turn)

        if not actions:
            if not any(can_put(board, row, col, not is_black_turn) for row in range(8) for col in range(8)):
                break  # 両者とも合法手がないなら終了
            is_black_turn = not is_black_turn
            continue

        action = choose_action_dqn(model, state, epsilon, len(actions), is_black_turn)


        # 行動を実行
        put(board, action[0], action[1], is_black_turn)

        black_count, white_count = count_stones(board)
        reward = 0
        if black_count + white_count == 64 or not any(can_put(board, row, col, not is_black_turn) for row in range(8) for col in range(8)):
            reward = 1 if black_count > white_count else -1 if black_count < white_count else 0
        total_reward += reward

        next_state = get_state_str(board)
        done = (black_count + white_count == 64) or not any(can_put(board, row, col, not is_black_turn) for row in range(8) for col in range(8))

        # 経験をバッファに追加
        replay_buffer.add((state, action, reward, next_state, done))

        # Qネットワークの更新
        update_dqn_model(model, replay_buffer, batch_size, gamma)

        is_black_turn = not is_black_turn

    return total_reward

# DQNの設定
input_shape = (8, 8)  # 盤面の形状
num_actions = 64  # 盤面の各セルを1つのアクションとして扱う
model = create_dqn_model(input_shape, num_actions)
replay_buffer = ReplayBuffer(max_size=10000)
epsilon = 1.0  # 初期epsilon
epsilon_decay = 0.995  # epsilon減衰
epsilon_min = 0.01
batch_size = 32
alpha = 0.1
gamma = 0.9
episodes = 1000

rewards_log = []

for episode in range(episodes):
    reward = play_game_dqn(model, replay_buffer, batch_size, alpha, gamma, epsilon)
    rewards_log.append(reward)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1} finished, Reward: {reward}")
        print(f"終わりました")

    # epsilonを減衰
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if (episode + 1) % 100 == 0:
        if not os.path.exists('saved_model'):
            os.makedirs('saved_model')
        model.save('saved_model/dqn1000_model')

# 学習の進捗をグラフ化（matplotlibを使用）
import matplotlib.pyplot as plt

plt.plot(rewards_log)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Progress")
plt.show()
