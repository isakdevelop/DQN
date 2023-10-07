import random
import torch
import torch.nn as nn
import torch.optim as optim

# 신경망 아키텍처 정의
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN 클래스 정의
class DQN:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.q_network = QNetwork(input_size, output_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

    def select_action(self, state):
        q_values = self.q_network(state)
        action_probs = nn.functional.softmax(q_values, dim=-1)
        action = torch.multinomial(action_probs, 1).item()
        return action

    def train(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(1)
        actions = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        current_q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * next_q_values * gamma

        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 환경 및 매개변수 설정
actions = [0, 1]  # 0: 씻음, 1: 버림
num_actions = len(actions)
gamma = 0.99  # 할인 인자
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN(input_size=1, output_size=num_actions)

# 학습 반복
num_episodes = 1000
for episode in range(num_episodes):
    state = random.choice(["덜자람", "정상", "병에 걸림", "부서짐"])  # 모든 상태 선택
    state_idx = 0 if state == "덜자람" or state == "정상" else 1

    done = False
    while not done:
        state_tensor = torch.tensor([state_idx], dtype=torch.float32, device=device).unsqueeze(1)
        action = dqn.select_action(state_tensor)

        if state == "덜자람" or state == "정상":
            next_state = "씻음"
            reward = 1 if action == 0 else -1  # 올바른 선택 시 보상 1, 잘못된 선택 시 보상 -1
            next_state_idx = 0 if next_state == "씻음" else 1
        else:
            next_state = "버림"
            reward = 1 if action == 1 else -1  # 올바른 선택 시 보상 1, 잘못된 선택 시 보상 -1
            next_state_idx = 1 if next_state == "버림" else 0

        dqn.train([(state_idx, action, reward, next_state_idx, done)])

        done = True  # 항상 끝나도록 설정
        state_idx = next_state_idx

        if action == 0:
            print(f"에피소드 {episode + 1}: 상태 = {state}, 액션 = 씻음")
        else:
            print(f"에피소드 {episode + 1}: 상태 = {state}, 액션 = 버림")
print("학습 완료")