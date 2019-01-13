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
            for h in self.hidden: h.cuda()
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
        return F.softmax(probs), value

    @staticmethod
    def preprocess(image):
        image = np.array(image)
        image = image / 255.0
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0).float()
        return image

class A2C():
    def __init__(self, env, learning_rate=1e-4, seq_len=10, reward_scaling=1,
                 discount_factor=0.99, grad_clip=1,
                 actor_coeff=1.0, critic_coeff=0.5, entropy_coeff=0.001):
        self.env       = env
        self.model     = MiniGridNet(env.action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

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
        total_score = 0
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        for episode in range(num_episodes):
            logging = {
                'score': 0,
                'actor_loss': 0,
                'critic_loss': 0,
                'entropy_loss': 0
            }
            num_steps = 0
            self.model.reset_hidden()
            obs    = self.env.reset()
            done   = False
            seq_no = 0
            log_probs = []
            values = []
            rewards = []
            entropy = 0
            while not done:
                obs = self.preprocess(obs["image"])
                probs, value = self.model(obs)

                dist = Categorical(probs)
                # Sample action from probs for exploration
                action = dist.sample()
                self.env.render('human')
                obs, reward, done, _ = self.env.step(action)
                reward = reward / self.reward_scaling

                log_probs.append(dist.log_prob(action))
                entropy += dist.entropy().mean()
                values.append(value)
                rewards.append(reward)

                seq_no += 1
                num_steps += 1
                if seq_no > self.seq_len or done:
                    # Back prop if seq_len reached or goal reached
                    # NOTE: Ignoring one state in unrolling for simplicity
                    #       If any weird training error, try fixing this.
                    if done:
                        R = 0
                    else:
                        _, value = self.model(self.preprocess(obs["image"]), retain_hidden=True)
                        R = value.item()
                    R = torch.FloatTensor([[R]])
                    returns = []
                    for i in reversed(range(len(rewards))):
                        R = rewards[i] + (self.discount_factor * R)
                        returns.insert(0, R)

                    if len(returns) == 0:
                        break

                    # Convert list of tensors [(1, n)] ->  tensor (n)
                    log_probs = torch.cat(log_probs)
                    returns   = torch.cat(returns).detach()
                    values    = torch.cat(values)
                    advantage = returns - values

                    # -1 in actor because gradient ascent
                    actor_loss  = -1 * (log_probs * advantage.detach()).mean()
                    critic_loss = advantage.pow(2).mean()

                    loss  = 0
                    loss += (self.actor_coeff * actor_loss)
                    loss += (self.critic_coeff * critic_loss)
                    loss -= (self.entropy_coeff * entropy)

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                    logging['score'] += torch.Tensor(rewards).sum()
                    logging['actor_loss'] += actor_loss.item()
                    logging['critic_loss'] += critic_loss.item()
                    logging['entropy_loss'] += entropy.item()

                    self.model.detach()
                    seq_no = 0
                    log_probs = []
                    values  = []
                    rewards = []
                    entropy = 0
                    # if done, next episode
                    if done:
                        break
            print("Episode[%d/%d] [Steps: %d]: [Score: %.2f] [Loss(A,C,E): %.3f %.3f %.3f]"%
                  (episode+1,
                   num_episodes,
                   num_steps,
                   logging['score'],
                   logging['actor_loss'], logging['critic_loss'], logging['entropy_loss'])
            )
            total_score += logging['score']
            total_actor_loss += logging['actor_loss']
            total_critic_loss += logging['critic_loss']
            total_entropy_loss += logging['entropy_loss']

        total_score /= num_episodes
        total_actor_loss /= num_episodes
        total_critic_loss /= num_episodes
        total_entropy_loss /= num_episodes

        print("Train[%d episodes]: [Score: %.2f] [Loss(A,C,E): %.3f %.3f %.3f]"%
              (num_episodes, total_score, total_actor_loss, total_critic_loss, total_entropy_loss))

    def val(self, num_episodes=100):
        pass

if __name__ == "__main__":
    # Test code
    model = MiniGridNet()
    inp = torch.zeros(1, 3, 7, 7)
    out = model(inp)
    print(out.shape)
