import torch

from core.models.vin.vin_base import VINBase


class CALVINBase(VINBase):
    def get_motion_model(self):
        raise NotImplementedError

    def get_available_actions(self, input_view, motion_model, target_map):
        """
        :param input_view: (batch_sz, l_i, *state_shape)
        :return:
            available actions A(s, a): (batch_sz, n_actions, *state_shape)
            available actions logit: (batch_sz, n_actions, *state_shape) or None
            available actions thresh: (batch_sz, n_actions, *state_shape) or None
        """
        raise NotImplementedError

    def get_target_map(self, feature_map, target=None):
        return None

    def get_reward_function(self, feature_map, available_actions, motion_model):
        """
        :param available_actions: (batch_sz, n_actions, *state_shape)
        :param motion_model: (n_actions, *state_shape)
        :param reward_map: (batch_sz, 1, *state_shape)
        :return: reward function R(s, a): (batch_sz, n_actions, *state_shape)
        """
        raise NotImplementedError

    def eval_q(self, available_actions, motion_model, reward, value=None):
        raise NotImplementedError

    def _forward(self, feature_map=None, k=None, prev_v=None, inference=False, target=None, **kwargs):
        """
        :param feature_map: (batch_sz, imsize, imsize)
        :param k: number of iterations. If None, use config.k
        :param prev_v: previously evaluated v (if it exists)
        :return: logits and softmaxed logits
        """
        # get reward map
        motion_model = self.get_motion_model() #.detach()
        # get target map
        target_map = self.get_target_map(feature_map, target=target)
        # get probability of available actions
        aa, aa_logit, aa_thresh, free, free_logit = self.get_available_actions(feature_map, motion_model, target_map)
        # get reward function
        r = self.get_reward_function(feature_map, aa, motion_model)

        q = self.eval_q(aa, motion_model, r, prev_v)  # Initial Q value from reward
        v, _ = torch.max(q, dim=1, keepdim=True)

        # Update q and v values
        if k is None: k = self.kr if inference and self.kr else self.k
        for i in range(k):
            q = self.eval_q(aa, motion_model, r, v)
            v, _ = torch.max(q, dim=1, keepdim=True)

        results = {"q": q, "v": v, "prev_v": prev_v if prev_v is not None else torch.zeros_like(v),
                "r_sa": r, "r": r[:, self.actions.done_index],
                "aa": aa, "aa_logit": aa_logit, "aa_thresh": aa_thresh, "mm": motion_model}
        if free is not None:
            results['free'] = free
        if free_logit is not None:
            results['free_logit'] = free_logit
        if target_map is not None:
            results['target_map'] = target_map
        return results
