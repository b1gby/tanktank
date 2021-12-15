#!/usr/bin/env python
# coding: utf-8

# In[1]:


## connect Google Drive
#from google.colab import drive
#drive.mount('/content/drive') #, force_remount=True
#model_dir = '/content/drive/MyDrive/tank_model/'
# !ls /content/drive/MyDrive/tank_model/


# ## Attention Please
# requirements of using colab to train DQN
# * add this code
# * using GPU
# 
# -----
# 
# customed DQN 
# * self-modifed env is needed

# ## Self-defined Gym env
# 
# * https://blog.csdn.net/extremebingo/article/details/80867486
# * https://zhuanlan.zhihu.com/p/102920005
# 
# 
# an answer about how pygame works with gym
# 
# * https://stackoverflow.com/questions/58974034/pygame-and-open-ai-implementation
# 
# offical tutorial
# 
# * https://github.com/openai/gym/blob/master/docs/creating_environments.md
# 
# case example
# 
# * https://github.com/AGKhalil/BlockDude_CL/blob/master/gym_blockdude/envs/blockdude_env.py
# 
# finally, how to use gym env
# 
# * https://colab.research.google.com/drive/1ioIri83vP7SgNAifZh7a1CYcsuvFnziK

# In[2]:


## code for running gym on colab
# !apt-get install -y xvfb python-opengl > /dev/null 2>&1
# !pip install gym pyvirtualdisplay > /dev/null 2>&1
# !pip install pygame
import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
display = Display(visible=0, size=(600, 400))
display.start()


# In[3]:


# %%capture
# !rm -r tanktank
# !git clone https://github.com/b1gby/tanktank.git
get_ipython().system('pip install -e tanktank')


# In[4]:


import pygame


# ## Train DQN after ENV is ready

# In[5]:


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
get_ipython().run_line_magic('pwd', '')
get_ipython().run_line_magic('cd', 'tanktank')
import gym_tank

get_ipython().run_line_magic('cd', 'gym_tank/envs')
env = gym.make('Tank-v0')#.unwrapped

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[6]:


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# In[7]:


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.bn5 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))))
        linear_input_size = convw * convh * 64

        self.fc1_adv = nn.Linear(in_features=linear_input_size, out_features=512)
        self.fc1_val = nn.Linear(in_features=linear_input_size, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=6)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        # self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.leaky_relu(self.bn1(self.conv1(x)),0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)),0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)),0.01)
        # x = self.maxpool(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.01)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.01)

        x = x.view(x.size(0), -1)

        adv = F.leaky_relu(self.fc1_adv(x), 0.01)
        val = F.leaky_relu(self.fc1_val(x), 0.01)

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), 6)
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), 6)
        return x


# In[8]:


resize = T.Compose([T.ToPILImage(),
                    T.Resize(400, interpolation=Image.CUBIC),
                    T.ToTensor()])


# def get_cart_location(screen_width):
#     world_width = env.x_threshold * 2
#     scale = screen_width / world_width
#     return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
        # _, screen_height, screen_width = screen.shape
        # screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        # view_width = int(screen_width * 0.6)
        # cart_location = get_cart_location(screen_width)
        # if cart_location < view_width // 2:
        #     slice_range = slice(view_width)
        # elif cart_location > (screen_width - view_width // 2):
        #     slice_range = slice(-view_width, None)
        # else:
        #     slice_range = slice(cart_location - view_width // 2,
        #                         cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
        # screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

    screen = torch.from_numpy(screen)
    # print('inner shape',screen.shape)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


env.reset()
plt.figure()
cur_screen = get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy()
print(cur_screen.shape)
plt.imshow(cur_screen,
           interpolation='none')
plt.title('Example extracted screen')
plt.show()


# In[9]:


BATCH_SIZE = 20 #128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
MEMORY_CAPACITY = 200 ## 10000

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

print('n_actions')

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(MEMORY_CAPACITY)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) *         math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    ACTION = None
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            ACTION =  policy_net(state).max(1)[1].view(1, 1)
    else:
        ACTION =  torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    # print('tanking action: ', ACTION)
    return ACTION


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    
    print('episode_durations: ',episode_durations)

    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    # plt.savefig('/content/drive/MyDrive/tank_figure/' + str(random.randint(5,99999))+'.png')

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# In[10]:


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    print('---------->',loss.item())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# In[ ]:


model_dir = './mdoel'

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict( torch.load(model_dir + "dueling_dqn_target_net.pth"))     #  policy_net.state_dict()


policy_net.load_state_dict( torch.load(model_dir + "dueling_dqn_policy_net.pth"))     #  policy_net.state_dict()

import time
num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    # last_screen = get_screen()
    current_screen = get_screen()
    # plt.figure()
    # plt.imshow(current_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
    #        interpolation='none')
    # plt.show()

    sum_reward = 0
    state = current_screen #torch.cat((last_screen,current_screen), 0) #current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, info = env.step(action.item())
        
        sum_reward += reward
        # for i in info["green_bullets"]:
        #     print(i.x, i.y)
        # print('reward:', reward)
        # print('done: ',done)
        reward = torch.tensor([reward], device=device)
        
        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen #- last_screen
            plt.figure()
            plt.imshow(current_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),
                interpolation='none')
            plt.show()

        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    print(sum_reward)
    # print(policy_net.parameters)
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        torch.save(target_net.state_dict(), model_dir + "dueling_dqn_target_net.pth")
        torch.save(policy_net.state_dict(), model_dir + "dueling_dqn_policy_net.pth")
    time.sleep(2)
    
print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()


# In[ ]:





# In[ ]:




