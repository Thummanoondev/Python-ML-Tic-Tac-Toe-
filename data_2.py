import numpy as np
import random

# สร้างเกมส์ Tic-Tac-Toe ขนาด 3x3
board = np.full((3, 3), '')  # ใช้ full ในการสร้างเกมส์เปล่าทั้งหมด

# อัพเดต Q-value ตามสมการ Q-learning
def update_q_value(Q, state, action, reward, next_state, alpha, gamma):
    max_next_q_value = np.max(Q[next_state, :])
    Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * max_next_q_value)

# Q-learning Algorithm
def q_learning(board, alpha, gamma, num_episodes):
    Q = np.zeros((3**9, 9))  # Q-value สำหรับแต่ละสถานการณ์และการกระทำ
    for episode in range(num_episodes):
        state = board_to_state(board)
        while True:
            # เลือกการกระทำที่สุ่ม
            action = random.choice(available_actions(board))
            # ทำการกระทำและได้รับรางวัล
            next_board = take_action(board, action, 'X')
            reward = calculate_reward(next_board)
            next_state = board_to_state(next_board)
            # อัพเดต Q-value โดยใช้สมการ Q-learning
            update_q_value(Q, state, action, reward, next_state, alpha, gamma)
            # เลื่อนไปสถานการณ์ถัดไป
            state = next_state
            board = next_board
            if game_over(board):
                break
    return Q

# แปลงสถานการณ์จากเกมส์ Tic-Tac-Toe เป็นสถานการณ์ใน Q-learning
# แปลงสถานการณ์จากเกมส์ Tic-Tac-Toe เป็นตัวเลข
def board_to_state(board):
    state = 0
    multiplier = 1
    for i in range(3):
        for j in range(3):
            if board[i, j] == 'X':
                state += multiplier * 1
            elif board[i, j] == 'O':
                state += multiplier * 2
            multiplier *= 3
    return state

# คำนวณรางวัลจากสถานการณ์ในเกมส์ Tic-Tac-Toe
def calculate_reward(board):
    if check_winner(board, 'X'):
        return 1
    elif check_winner(board, 'O'):
        return -1
    else:
        return 0

# ตรวจสอบว่ามีผู้ชนะในเกมส์ Tic-Tac-Toe หรือไม่
def check_winner(board, player):
    # ตรวจสอบแนวตั้งและแนวนอน
    for i in range(3):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return True
    # ตรวจสอบแนวทะแยง
    if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
        return True
    return False

# ตรวจสอบว่าเกมส์ Tic-Tac-Toe จบแล้วหรือไม่
def game_over(board):
    return check_winner(board, 'X') or check_winner(board, 'O') or not available_actions(board)

# คำนวณหมายเลขของช่องว่างในเกมส์ Tic-Tac-Toe
def available_actions(board):
    actions = [(i, j) for i in range(3) for j in range(3) if board[i, j] == '']
    if actions:  # ตรวจสอบว่ามีตำแหน่งที่สามารถเลือกได้หรือไม่
        return actions  # ถ้ามี ส่งคืนลิสต์ของตำแหน่งที่สามารถเลือกได้
    else:
        return [(0, 0)]  # ถ้าไม่มี ส่งคืนตำแหน่งแรกเพื่อหลีกเลี่ยง IndexError


# ทำการกระทำในเกมส์ Tic-Tac-Toe
def take_action(board, action, player):
    if board[action] == '':
        next_board = board.copy()
        next_board[action] = player
        return next_board
    else:
        print("Invalid move. Position is already occupied.")
        return board



# เลือกการกระทำโดยใช้ Q-value
def choose_action(board, Q):
    state = board_to_state(board)
    if np.random.rand() < 0.5:
        # เลือกตาม Q-value
        action_idx = np.argmax(Q[state])
    else:
        # สุ่มเลือกจากตำแหน่งที่ว่าง
        action_idx = random.choice(range(9))  # เลือกเลขจาก 0 ถึง 8
    # แปลงจากตำแหน่งในรูปแบบเลข 0-8 เป็นคู่ของตำแหน่งในเกมส์
    action = (action_idx // 3, action_idx % 3)
    return action


# เล่นเกมส์ Tic-Tac-Toe ด้วย Q-learning
def play_game(Q1, Q2):
    round_count = 0  # เพิ่มตัวแปร round_count เพื่อนับรอบเกม
    board = np.full((3, 3), '')  # ใช้ full ในการสร้างเกมส์เปล่าทั้งหมด
    players = ['X', 'O']
    current_player = 0
    while not game_over(board):
        round_count += 1  # เพิ่มค่าของ round_count ทุกรอบ
        print(f"Round: {round_count}")  # แสดงรอบใน output
        print("Current board:")
        print(board)
        action = choose_action(board, Q1 if current_player == 0 else Q2)
        print(f"Player {players[current_player]} chooses position {action}")
        board = take_action(board, action, players[current_player])
        current_player = 1 - current_player

    print("Final board:")
    print(board)
    winner = 'X' if check_winner(board, 'X') else 'O' if check_winner(board, 'O') else None
    if winner:
        print(f"Player {winner} wins!")
    else:
        print("It's a draw!")

        
# การเรียนรู้ Q-value สำหรับทั้งสองบอท
Q1 = q_learning(board, alpha=0.2, gamma=0.9, num_episodes=10000)
Q2 = q_learning(board, alpha=0.1, gamma=0.9, num_episodes=10000)
# เริ่มเล่นเกม
play_game(Q1, Q2)
