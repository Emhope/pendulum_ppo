import torch
from torch.utils.tensorboard import SummaryWriter
from enum import Enum
from dataclasses import dataclass
from typing import Generator
import numpy as np
import os

from config import cfg
from env import InvertedPendulumEnv


class GetActionMode(Enum):
    train = "TRAIN"
    evaluate = "EVALUATE"

@dataclass
class PPOBuffer:
    states: torch.Tensor
    actions: torch.Tensor
    returns: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    
    def get_batches(self, batch_size, shuffle=True) -> Generator['PPOBuffer', None, None]:
        buffer_len = self.states.shape[0]
        idxs = np.arange(buffer_len)
        if shuffle:
            np.random.shuffle(idxs)
        for i in range(0, buffer_len, batch_size):
            batch_slice = idxs[i: min(buffer_len, i+batch_size)]
            yield PPOBuffer(
                states=self.states[batch_slice],
                actions=self.actions[batch_slice],
                returns=self.returns[batch_slice],
                log_probs=self.log_probs[batch_slice],
                advantages=self.advantages[batch_slice],
            )


class Actor(torch.torch.nn.Module):
    def __init__(self, state_dim, action_dim, net_width=64):
        super().__init__()
        self.base = torch.nn.Sequential(
            torch.nn.Linear(state_dim, net_width),
            torch.nn.Tanh(),
            torch.nn.Linear(net_width, net_width),
            torch.nn.Tanh(),
        )
        self.mu = torch.nn.Sequential(
            torch.nn.Linear(net_width, action_dim),
            torch.nn.Tanh(),
        )
        self.sigma = torch.nn.Sequential(
            torch.nn.Linear(net_width, action_dim),
            torch.nn.Softplus(),
        )

    def forward(self, x):
        base_out = self.base(x)
        mu = self.mu(base_out)
        sigma = self.sigma(base_out)
        return mu, sigma

class Critic(torch.torch.nn.Module):
    def __init__(self, state_dim, net_width=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, net_width),
            torch.nn.Tanh(),
            torch.nn.Linear(net_width, net_width),
            torch.nn.Tanh(),
            torch.nn.Linear(net_width, 1),
        )

    def forward(self, x):
        out = self.net(x)
        return out


def compute_returns(rewards, masks, gamma=0.99):
    returns = []
    R = 0
    # Проходим траекторию в обратном порядке
    for reward, mask in zip(reversed(rewards), reversed(masks)):
        R = reward + gamma * mask * R
        returns.insert(0, R)  # Добавляем в начало списка
    return torch.tensor(returns, dtype=torch.float32)

def compute_gae(next_value, rewards, masks, values, gamma, lambd):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] * masks[step] - values[step]
        gae = delta + gamma * lambd * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns



class Agent:
    def __init__(self, state_dim, action_dim, max_act):
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            net_width=cfg.net_width
        )
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic = Critic(
            state_dim=state_dim,
            net_width=cfg.net_width
        )
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.max_act = max_act
        self.tb_writer = SummaryWriter()
        self.actor_steps = 0
        self.critic_steps = 0
        self.evals = 0
        self.best_val = -np.inf

    
    def adapt_action(self, action):
        """
        обрезка и преобразование действия из диапазона [-1, 1], в диапазон [-max_act, +max_act]
        """
        return torch.clip(action, -1, 1) * self.max_act

    def get_raw_action(self, state, mode: GetActionMode = GetActionMode.train):
        """
        если мод evaluate, то выдается матож, без логпроба (вместо него None)
        иначе выдается сэмплированное действие и логпроб
        """
        with torch.no_grad():
            mu, sigma = self.actor(state)
            if mode == GetActionMode.evaluate:
                return mu, None
            elif mode == GetActionMode.train:
                dist = torch.distributions.Normal(mu, sigma)
                action = dist.sample()
                logprob = dist.log_prob(action)
                return action, logprob
    
    def update(self, buffer: PPOBuffer):
        for _ in range(cfg.train_actor_times):
            for batch in buffer.get_batches(batch_size=cfg.batch_size):  
                mu, std = self.actor(batch.states)
                dist = torch.distributions.Normal(mu, std)
                new_log_probs = dist.log_prob(batch.actions)
                entropy = dist.entropy().mean()
                ratio = (new_log_probs - batch.log_probs).exp()
                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * batch.advantages
                actor_loss = -torch.min(surr1, surr2).mean() - cfg.entropy_bonus * entropy
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()
                
                self.tb_writer.add_scalar('loss/actor', actor_loss, self.actor_steps)
                self.tb_writer.add_scalar('entropy', entropy, self.actor_steps)
                self.actor_steps += 1

        for _ in range(cfg.train_critic_times):
            for batch in buffer.get_batches(batch_size=cfg.batch_size):  
                new_vals = self.critic(batch.states)
                critic_loss = (new_vals.squeeze() - batch.returns).pow(2).mean()
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

                self.tb_writer.add_scalar('loss/critic', critic_loss, self.critic_steps)
                self.critic_steps += 1

    def train(self, env: InvertedPendulumEnv):
        for e in range(cfg.epochs):
            states, actions, rewards, masks, values, log_probs = [], [], [], [], [], []
            steps = 0
            state = env.reset_model()
            done = False
            total_reward = 0
            while steps < cfg.epoch_steps:
                steps += 1
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    value = self.critic(state_tensor)
                action, log_prob = self.get_raw_action(state_tensor)
                real_action = self.adapt_action(action)
                next_state, reward, done = env.step(real_action.numpy()[0])
                
                states.append(state)
                actions.append(action.numpy()[0])
                rewards.append(reward)
                masks.append(1 - done)
                values.append(value.item())
                log_probs.append(log_prob)
                
                state = next_state
                total_reward += reward
                
                if done:
                    state = env.reset_model()
            
            print(f'Iteration: {e}, Total Reward: {total_reward}')
            self.tb_writer.add_scalar('reward', total_reward, e)
            total_reward = 0

            # Вычисление преимуществ
            next_state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                next_value = self.critic(next_state_tensor)
            returns = compute_gae(next_value.item(), rewards, masks, values, cfg.gamma, cfg.gae_lambda)
            # Преобразование в тензоры
            states_tensor = torch.FloatTensor(np.array(states))
            actions_tensor = torch.FloatTensor(np.array(actions))
            returns_tensor = torch.FloatTensor(returns)
            log_probs_tensor = torch.cat(log_probs)
            values_tensor = torch.FloatTensor(values)
            advantages = returns_tensor - values_tensor
            # Нормализация преимуществ
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
 
            buffer = PPOBuffer(
                states=states_tensor,
                actions=actions_tensor,
                returns=returns_tensor,
                log_probs=log_probs_tensor,
                advantages=advantages,
            )
            self.update(buffer)
            cfg.entropy_bonus *= cfg.entropy_decay
            if e % cfg.validate_freq == 0:
                # можно было бы несколько раз валидировать и брать среднее, но 
                # начальное положение среды и действия актора детерминированы,
                # поэтому все попытки будут одинаковыми
                v = self.eval_traj(env)
                self.tb_writer.add_scalar('validation', v, e)
                if v > self.best_val:
                    try:
                        os.remove(f'models//best_model{self.best_val}.pth')
                    except:
                        pass
                    self.best_val = v
                    print(f'new best model at {e} iteration')
                    self.save_actor(f'best_model{self.best_val}')
            if e % cfg.save_freq == 0:
                self.save_actor(f'actor{e}')


    def eval_traj(self, env: InvertedPendulumEnv):
        state = env.reset_model()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            act, _ = self.get_raw_action(state_tensor, mode=GetActionMode.evaluate)
            real_act = self.adapt_action(act)
            state, reward, done = env.step(real_act)
            total_reward += reward
        return total_reward

    def save_actor(self, name, dir='.//models'):
        torch.save(self.actor.state_dict(), f'{dir}//{name}.pth')
    
    def load_actor(self, name):
        self.actor.load_state_dict(torch.load(name))


if __name__ == '__main__':
    env = InvertedPendulumEnv()
    agent = Agent(4, 1, 3)
    agent.train(env)
