from dataclasses import dataclass


@dataclass
class LearnConfig:
    batch_size: int
    epochs: int
    epoch_steps: int
    train_actor_times: int
    train_critic_times: int
    entropy_bonus: float
    entropy_decay: float
    gamma: float # discount factor
    gae_lambda: float
    clip_epsilon: float
    actor_lr: float
    critic_lr: float
    net_width: int
    save_freq: int
    validate_freq: int


cfg = LearnConfig(
    batch_size=256,
    epochs=1_000,
    epoch_steps=2048,
    train_actor_times=10,
    train_critic_times=10,
    entropy_bonus=0.01,
    entropy_decay=0.99,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    actor_lr=2e-4,
    critic_lr=2e-4,
    net_width=150,
    save_freq=50,
    validate_freq=1,
)
