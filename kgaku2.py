import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import pickle  # Qテーブル保存用

# Qテーブルの初期化
Q_table = {}

# Qテーブルの保存
def save_q_table(filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(Q_table, f)

# Qテーブルの読み込み
def load_q_table(filename="q_table.pkl"):
    global Q_table
    try:
        with open(filename, "rb") as f:
            Q_table = pickle.load(f)
        print("Qテーブルを読み込みました。")
    except FileNotFoundError:
        print("Qテーブルが見つかりませんでした。新しく作成します。")



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

# Q学習用のQテーブル
Q_table = {}

def get_state_str(board):
    return str(board.flatten())

def choose_action(state, actions, epsilon):
    if random.random() < epsilon:
        return random.choice(actions)  # ランダムに選択
    else:
        q_values = [Q_table.get((state, action), 0) for action in actions]
        return actions[np.argmax(q_values)]

def update_q_table(state, action, reward, next_state, is_black_turn, alpha, gamma):
    next_board = get_board_from_state_str(next_state)  # 文字列から盤面に変換
    next_actions = get_valid_moves(next_board, is_black_turn)
    
    max_next_q = max([Q_table.get((next_state, a), 0) for a in next_actions]) if next_actions else 0
    
    Q_table[(state, action)] = Q_table.get((state, action), 0) + \
        alpha * (
            reward + gamma * max_next_q - Q_table.get((state, action), 0)
        )

    
# 状態文字列を元の盤面（NumPy配列）に戻す関数
def get_board_from_state_str(state_str):
    state_array = np.fromstring(state_str.strip("[]"), sep=" ", dtype=int)
    return state_array.reshape((8, 8))

def update_q_table(state, action, reward, next_state, is_black_turn, alpha, gamma):
    next_board = get_board_from_state_str(next_state)  # 文字列から盤面に変換
    next_actions = get_valid_moves(next_board, is_black_turn)  # 次の状態での合法手を取得

    # 合法手がない場合のデフォルト値を設定
    max_next_q = max([Q_table.get((next_state, a), 0) for a in next_actions]) if next_actions else 0

    # Qテーブルの更新
    Q_table[(state, action)] = Q_table.get((state, action), 0) + \
        alpha * (
            reward + gamma * max_next_q - Q_table.get((state, action), 0)
        )
    
def play_game_q_learning(alpha=0.1, gamma=0.9, epsilon=0.1, log_rewards=False):
    board = init_board()
    is_black_turn = True
    total_reward = 0  # 報酬の合計を記録

    while True:
        state = get_state_str(board)
        actions = get_valid_moves(board, is_black_turn)

        if not actions:  # 合法手がない場合
            if not any(can_put(board, row, col, not is_black_turn) for row in range(8) for col in range(8)):
                break  # 両者とも合法手がないなら終了
            is_black_turn = not is_black_turn
            continue

        # 行動を選択
        action = choose_action(state, actions, epsilon)

        # 行動を実行
        put(board, action[0], action[1], is_black_turn)

        # 報酬を計算（ゲーム終了時の勝利判定に基づく報酬）
        black_count, white_count = count_stones(board)
        reward = 0
        if black_count + white_count == 64 or not any(can_put(board, row, col, not is_black_turn) for row in range(8) for col in range(8)):
            reward = 1 if black_count > white_count else -1 if black_count < white_count else 0
        total_reward += reward

        # 次の状態
        next_state = get_state_str(board)

        # Qテーブルを更新
        update_q_table(state, action, reward, next_state, is_black_turn, alpha, gamma)

        # ターン切り替え
        is_black_turn = not is_black_turn

    return total_reward

# 学習ループ
load_q_table()  # 過去のQテーブルをロード
rewards_log = []

for episode in range(10000):
    reward = play_game_q_learning()
    rewards_log.append(reward)

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1} finished, Reward: {reward}")
        save_q_table()  # 定期的にQテーブルを保存

# 学習の進捗をグラフ化（matplotlibを使用）
import matplotlib.pyplot as plt

plt.plot(rewards_log)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Training Progress")
plt.show()


