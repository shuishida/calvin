import torch

from core.model import Model
from core.mdp.actions import ActionSetBase


class VINBase(Model):
    def __init__(self, *, action_set: ActionSetBase = None, device="cuda", k=None, kr=None, gamma=0.99, l_i=None,
                 discount=None, dropout=None, sparse=False, dense=False, softmax=None, epsilon=0.05, target_known=False, prev_state=False, **kwargs):
        super(VINBase, self).__init__()
        self.device = device
        self.k = k
        self.kr = kr
        self.gamma = gamma
        self.actions = action_set
        self.discount = discount
        self.dropout = dropout
        self.sparse = sparse
        self.dense = dense
        self.sm_temperature = softmax
        self.epsilon = epsilon
        self.target_known = target_known
        self.prev_state = prev_state
        self.l_i = l_i + int(target_known) + int(prev_state)

    def __repr__(self):
        return f"{self.__class__.__name__}_k_{self.k}"

    def add_channel(self, feature_map, poses):
        channel_size = list(feature_map.size())
        channel_size[1] = 1
        channel = torch.zeros(channel_size, device=self.device)

        assert poses is not None
        poses = poses.long()
        channel[torch.arange(len(poses)), -1, poses[:, 0], poses[:, 1]] = 1
        channel[:, -1, 0, 0] = 0
        return torch.cat([feature_map, channel], dim=1)

    def forward(self, feature_map=None, prev_v=None, prev_poses=None, new_episodes=None, target=None, **kwargs):
        """
        :param feature_map: (batch_sz, imsize, imsize)
        :param k: number of iterations. If None, use config.k
        :param prev_v: previously evaluated v (if it exists)
        :return: logits and softmaxed logits
        """
        if self.target_known:
            feature_map = self.add_channel(feature_map, target)

        if prev_v is not None:
            assert new_episodes is not None
            prev_v = prev_v.clone().detach()
            prev_v[new_episodes] = 0
        return self._forward(feature_map=feature_map, prev_v=prev_v, new_episodes=new_episodes, target=target, **kwargs)

    def _forward(self, **kwargs) -> dict:
        raise NotImplementedError

    def sample_actions(self, scores, explore=False, **kwargs):
        action_labels = scores.argmax(dim=1)
        if explore:
            batch_sz, n_actions = scores.size()
            action_labels = torch.where(torch.rand(batch_sz, device=self.device) > self.epsilon,
                                        action_labels,
                                        torch.randint(n_actions, (batch_sz,), device=self.device))
        elif self.sm_temperature:
            sm_scores = torch.softmax(scores * self.sm_temperature, dim=1)
            dist = torch.distributions.Categorical(sm_scores)
            action_labels = dist.sample()

        return action_labels

    def action(self, q=None, curr_poses=None, explore=False, softmax=False, **kwargs):
        pred_scores = self.extract_state_q(q, curr_poses.long())
        return self.sample_actions(pred_scores, explore=explore, softmax=softmax)

    def extract_state_q(self, q, state):
        raise NotImplementedError


def add_vin_args(parser):
    parser.add_argument('--k', type=int, required=True, help='Number of Value Iterations')
    parser.add_argument('--k_sz', type=int, default=3, help='Kernel size')
    parser.add_argument('--kr', type=int, default=None, help='Number of Value Iterations during rollout')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma for Value Iterations')
    parser.add_argument('--epsilon', type=float, default=0.05, help='Gamma for Value Iterations')
    parser.add_argument('--ideal_trans', '-tr', action='store_true', default=False,
                        help='assume ideal transition to set r_motion and w_motion')
    parser.add_argument('--use_policy_net', '-pn', action='store_true', default=False,
                        help='set if additional policy for mapping from q values to actions is desired')
    parser.add_argument('--l_i', type=int, default=3, help='Number of channels in input layer')
    parser.add_argument('--l_h', type=int, default=150, help='Number of channels in first hidden layer')
    parser.add_argument('--l_q', type=int, default=9, help='Number of channels in q layer (~actions) in VI-module')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of conv layers in VI-module')
    parser.add_argument('--motion_scale', "-ms", type=float, default=10.0)
    parser.add_argument('--w_loss_p', "-wlp", type=float, default=1.0)
    parser.add_argument('--w_loss_q', "-wlq", type=float, default=1.0)
    parser.add_argument('--w_loss_d', "-wld", type=float, default=1.0)
    parser.add_argument('--w_loss_optim', "-wlo", type=float, default=1.0)
    parser.add_argument('--sparse', '-sp', action='store_true', default=False,
                        help="evaluate best action only for current state")
    parser.add_argument('--dense', '-dns', action='store_true', default=False,
                        help="evaluate best action for all visible states")
    parser.add_argument('--discount', '-dis', type=float, default=0.0, help='Discount rate of distance')
    parser.add_argument('--dropout', '-drop', type=float, default=0.0, help='Dropout')
    parser.add_argument('--target_known', '-known', action='store_true', default=False)
    return parser
