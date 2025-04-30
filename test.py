from main import Agent, GetActionMode
from env import InvertedPendulumEnv

import torch
import mujoco.viewer
import time


agent = Agent(4, 1, 3)
agent.load_actor('models/best_model[331.9419155].pth')
env = InvertedPendulumEnv()
env.viewer = mujoco.viewer.launch_passive(env.model, env.data)
state = env.reset_model()
done = False
while True:
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    act, _ = agent.get_raw_action(state_tensor, mode=GetActionMode.evaluate)
    real_act = agent.adapt_action(act)
    state, reward, done = env.step(real_act)
    env.viewer.sync()
    time.sleep(0.02)
    if done:
        state = env.reset_model()

