import random
import numpy as np
import pickle  # Qテーブルの読み込み用

# Qテーブルのロード
def load_q_table(filename="q_table.pkl"):
    global Q_table
    try:
        with open(filename, "rb") as f:
            Q_table = pickle.load(f)
        print("Qテーブルを読み込みました。")
    except FileNotFoundError:
        print("Qテーブルが見つかりませんでした。")

# 盤面の状態（0: 空き, 1: 黒, -1: 白）
def init_board():
    board = np.zeros((8, 8), dtype=int)
    board[3, 3] = board[4, 4] = -1  # 白
    board[3, 4] = board[4, 3] = 1   # 黒
    return board

# 石を置けるか確認する関数
def can_put(board, row, col, is_black_turn):
    if board[row, col] != 0:
        return False

    current_player = 1 if is_black_turn else -1
    opponent_player = -1 if is_black_turn else 1
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for direction in directions:
        x, y = row + direction[0], col + direction[1]
        has_opponent_between = False

        while 0 <= x < 8 and 0 <= y < 8:
            if board[x, y] == opponent_player:
                has_opponent_between = True
            elif board[x, y] == current_player:
                if has_opponent_between:
                    return True
                else:
                    break
            else:
                break
            x += direction[0]
            y += direction[1]

    return False

# Q学習AIによる手を選ぶ関数
def q_learning_move(board, Q_table, is_black_turn, epsilon=0.1):
    valid_moves = [(row, col) for row in range(8) for col in range(8) if can_put(board, row, col, is_black_turn)]
    if valid_moves:
        state = str(board.flatten())  # 現在の状態を文字列に変換
        q_values = [Q_table.get((state, move), 0) for move in valid_moves]
        
        # epsilon-greedy方針で行動を選択
        if random.random() < epsilon:
            return random.choice(valid_moves)  # ランダムに選択
        else:
            return valid_moves[np.argmax(q_values)]  # 最大Q値の手を選択
    return None

# ランダムAIによる手を選ぶ関数
def random_move(board, is_black_turn):
    valid_moves = [(row, col) for row in range(8) for col in range(8) if can_put(board, row, col, is_black_turn)]
    return random.choice(valid_moves) if valid_moves else None

# 盤面に石を置き、石を反転する関数
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

# 盤面を表示する関数
def print_board(board):
    for row in board:
        print(" ".join(["●" if x == 1 else "○" if x == -1 else "." for x in row]))
    print()

# 勝者判定のために石の数をカウントする関数
def count_stones(board):
    black = np.sum(board == 1)
    white = np.sum(board == -1)
    return black, white

# ゲームのシミュレーション
def simulate_game(Q_table, num_games=100, epsilon=0.1):
    q_learning_wins = 0
    random_wins = 0
    draws = 0

    for i in range(num_games):
        board = init_board()
        is_black_turn = True

        while True:
            print_board(board)
            # 現在のプレイヤーの合法手をチェック
            if is_black_turn:
                move = random_move(board, is_black_turn)  # ランダムAI（黒）
            else:
                move = q_learning_move(board, Q_table, is_black_turn, epsilon)  # Q学習AI（白）

            if move:
                put(board, move[0], move[1], is_black_turn)
            else:
                # 両者とも合法手がない場合は終了
                if not any(can_put(board, row, col, not is_black_turn) for row in range(8) for col in range(8)):
                    break
            is_black_turn = not is_black_turn  # ターンを切り替え

        # ゲーム終了後の結果をカウント
        black, white = count_stones(board)
        if black > white:
            random_wins += 1  # 黒（ランダムAI）の勝利
        elif white > black:
            q_learning_wins += 1  # 白（Q学習AI）の勝利
        else:
            draws += 1

    return q_learning_wins, random_wins, draws

# Qテーブルのロード
load_q_table("q_table.pkl")

# シミュレーションを実行
num_simulations = 100
q_learning_wins, random_wins, draws = simulate_game(Q_table, num_simulations)

# 結果の表示
print(f"{num_simulations}回の対戦結果:")
print(f"ランダムAI（黒）の勝利数: {random_wins}")
print(f"Q学習AI（白）の勝利数: {q_learning_wins}")
print(f"引き分け: {draws}")
