import time
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.distributions import Categorical

class MiniGridNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        # Assuming input to network is 7x7
        self.base = nn.Sequential(
            # (1x3x7x7)
            nn.Conv2d( 3,  32, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            # (1x32x4x4)
            nn.Conv2d(32,  64, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            # (1x64x2x2)
            nn.Conv2d(64, 128, kernel_size=2),
            nn.ELU(inplace=True),
            # (1x128x1x1)
        )

        self.lstm   = nn.LSTMCell(128, 128)
        self.actor  = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

        self.hidden = None
        self.reset_hidden()

    def reset_hidden(self):
        self.hidden = [torch.zeros(1, 128), torch.zeros(1, 128)]
        if next(self.lstm.parameters()).is_cuda:
            self.hidden = [h.cuda() for h in self.hidden]
        return

    def detach(self):
        self.hidden = [h.detach() for h in self.hidden]
        return

    def forward(self, x, retain_hidden=False):
        old_hidden = self.hidden
        base_embedding = self.base(x)
        base_embedding = base_embedding.squeeze(-1).squeeze(-1)
        self.hidden = self.lstm(base_embedding, self.hidden)
        probs = self.actor(self.hidden[0])
        value = self.critic(self.hidden[0])
        if retain_hidden:
            self.hidden = old_hidden
        return F.softmax(probs, 1), value

    @staticmethod
    def preprocess(image, cuda=False):
        image = np.array(image)
        image = image / 255.0
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0).float()
        if cuda: image = image.cuda()
        return image

class A2C:
    def __init__(self, env, learning_rate=1e-3, seq_len=100, cuda=False,
                 reward_scaling=1, discount_factor=0.99, grad_clip=1,
                 actor_coeff=1.0, critic_coeff=0.5, entropy_coeff=0.001,
                 state_dict=None):
        self.env       = env
        self.model     = MiniGridNet(env.action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        if state_dict:
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optim'])

        if cuda:
            self.model = self.model.cuda()

        self.cuda            = cuda
        self.preprocess      = MiniGridNet.preprocess
        self.seq_len         = seq_len
        self.reward_scaling  = reward_scaling
        self.discount_factor = discount_factor
        self.grad_clip       = grad_clip
        self.actor_coeff     = actor_coeff
        self.critic_coeff    = critic_coeff
        self.entropy_coeff   = entropy_coeff


    def train(self, num_episodes=100):
        self.model.train()

        # Average info book-keeping
        avg_score        = 0
        avg_actor_loss   = 0
        avg_critic_loss  = 0
        avg_entropy_loss = 0
        avg_steps        = 0

        for episode in range(num_episodes):
            # Per episode book-keeping
            logging = {
                'score': 0,
                'actor_loss': 0,
                'critic_loss': 0,
                'entropy_loss': 0
            }

            self.model.reset_hidden()
            obs       = self.env.reset()
            num_steps = 0
            done      = False

            entropies = []
            log_probs = []
            values    = []
            rewards   = []
            while not done:
                # run model, get action and value
                _obs = self.preprocess(obs["image"], self.cuda)
                probs, value = self.model(_obs)

                # Choose action and step
                dist = Categorical(probs)
                action = dist.sample() # sample for exploration
                obs, reward, done, _ = self.env.step(action)
                reward = reward / self.reward_scaling

                # Record all required values
                entropies.append(dist.entropy())
                log_probs.append(dist.log_prob(action))
                values.append(value)
                rewards.append(reward)

                num_steps += 1

                if (num_steps % self.seq_len) == 0 or done:
                    # Back prop if seq_len reached or goal reached
                    if done:
                        R = 0
                    else:
                        _obs = self.preprocess(obs["image"], self.cuda)
                        _, value = self.model(_obs, retain_hidden=True)
                        R = value.item()
                    R = torch.FloatTensor([[R]])
                    if self.cuda: R = R.cuda()
                    returns = []
                    for i in reversed(range(len(rewards))):
                        R = rewards[i] + (self.discount_factor * R)
                        returns.insert(0, R)

                    if len(returns) == 0: break

                    # Convert list of tensors [(1, n)] ->  tensor (n)
                    log_probs = torch.cat(log_probs)
                    returns   = torch.cat(returns).detach()
                    values    = torch.cat(values)
                    entropies = torch.cat(entropies)
                    advantage = returns - values

                    # -1 in actor, entropy because gradient ascent
                    actor_loss  = -1 * (log_probs * advantage.detach())
                    logging['actor_loss'] += actor_loss.sum().item()
                    actor_loss = actor_loss.mean()

                    critic_loss = advantage.pow(2)
                    logging['critic_loss'] += critic_loss.sum().item()
                    critic_loss = critic_loss.mean()

                    entropy_loss = -1 * entropies
                    logging['entropy_loss'] += entropies.sum().item()
                    entropy_loss = entropy_loss.mean()

                    logging['score'] += torch.Tensor(rewards).sum()

                    loss  = 0
                    loss += (self.actor_coeff * actor_loss)
                    loss += (self.critic_coeff * critic_loss)
                    loss += (self.entropy_coeff * entropy_loss)

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                  self.grad_clip)
                    self.optimizer.step()

                    self.model.detach()
                    log_probs = []
                    values    = []
                    rewards   = []
                    entropies = []

                    # if done, next episode
                    if done:
                        break

            logging['actor_loss'] /= num_steps
            logging['critic_loss'] /= num_steps
            logging['entropy_loss'] /= num_steps

            print("Episode[%d/%d] [Steps: %d]: [Score: %.2f] [Loss(A,C,E): %.3f %.3f %.3f]"%
                  (episode+1,
                   num_episodes,
                   num_steps,
                   logging['score'],
                   logging['actor_loss'], logging['critic_loss'], logging['entropy_loss'])
            )

            avg_score        += logging['score']
            avg_actor_loss   += logging['actor_loss']
            avg_critic_loss  += logging['critic_loss']
            avg_entropy_loss += logging['entropy_loss']
            avg_steps        += num_steps

        avg_score        /= num_episodes
        avg_actor_loss   /= num_episodes
        avg_critic_loss  /= num_episodes
        avg_entropy_loss /= num_episodes
        avg_steps        /= num_episodes
        print("Train[%d episodes]: [Score: %.2f]: [Steps: %.2f]: [Loss(A,C,E): %.3f %.3f %.3f]"%
              (num_episodes, avg_score, avg_steps, avg_actor_loss, avg_critic_loss, avg_entropy_loss))

    @torch.no_grad()
    def val(self, num_episodes=100):
        self.model.eval()

        # Average info book-keeping
        avg_score        = 0
        avg_actor_loss   = 0
        avg_critic_loss  = 0
        avg_entropy_loss = 0
        avg_steps        = 0

        for episode in range(num_episodes):
            # Per episode book-keeping
            self.model.reset_hidden()
            obs       = self.env.reset()
            num_steps = 0
            done      = False

            score = 0
            while not done:
                # run model, get action and value
                _obs = self.preprocess(obs["image"], self.cuda)
                probs, value = self.model(_obs)

                # Choose action and step
                dist = Categorical(probs)
                action = dist.sample() # sample for exploration
                obs, reward, done, _ = self.env.step(action)
                reward = reward / self.reward_scaling

                # Record all required values
                score += reward
                num_steps += 1

            print("Episode[%d/%d] [Steps: %d]: [Score: %.2f]"%
                  (episode+1,
                   num_episodes,
                   num_steps,
                   score)
            )

            avg_score += score
            avg_steps += num_steps

        avg_score /= num_episodes
        avg_steps /= num_episodes
        print("Val[%d episodes]: [Score: %.2f]: [Steps: %.2f]"%(num_episodes, avg_score, avg_steps))

    @torch.no_grad()
    def visualize(self):
        self.model.eval()
        self.model.reset_hidden()
        obs = self.env.reset()
        while True:
            renderer = self.env.render('human')

            # run model, get action and value
            _obs = self.preprocess(obs["image"], self.cuda)
            probs, value = self.model(_obs)

            # Choose action and step
            dist = Categorical(probs)
            action = dist.sample() # sample for exploration
            obs, reward, done, _ = self.env.step(action)

            time.sleep(0.1)
            if renderer.window == None: break
            if done:
                obs = self.env.reset()
                self.model.reset_hidden()

    def checkpoint(self, step):
        torch.save(
            {
                'step' : step,
                'optim': self.optimizer.state_dict(),
                'model': self.model.state_dict()
            },
            'epoch-%d.save'%step
        )
