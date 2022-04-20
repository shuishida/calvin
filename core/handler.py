from collections import defaultdict
from typing import Any, Tuple, List

import torch

from core.utils.env_utils import NavStatus
from core.mdp.meta import MDPMeta


def add_prefix(data: dict, prefix: str):
    return {f"{prefix}{k}": v for k, v in data.items()}


def get_dict(data: dict, index=None, keys=None):
    if keys is None:
        keys = data.keys()
    elif not keys:
        return {}
    if index is None: return {k: data.get(k, []) for k in keys}
    return {k: data[k][index] for k in keys}


def eval_returns(rewards, gamma=1.0):
    discounts = gamma ** torch.arange(len(rewards))
    dis_rewards = rewards * discounts
    returns = dis_rewards.sum() * torch.ones_like(dis_rewards)
    if len(returns) > 1:
        returns[1:] -= torch.cumsum(dis_rewards[:-1], dim=0)
    return returns / discounts


def dict_collate(items_dict_list: List[dict], equal_length=True):
    items_list_dict = defaultdict(list)
    for items_dict in items_dict_list:
        for k, v in items_dict.items():
            items_list_dict[k].append(v)
    results = {}
    for k, v in items_list_dict.items():
        try:
            results[k] = (torch.stack if equal_length else torch.cat)(v) if isinstance(v[0], torch.Tensor) else torch.tensor(v)
        except:
            results[k] = v
    return results


class DataHandler:
    def __init__(self, meta: MDPMeta, gamma=1.0, require_returns=False, **kwargs):
        self.meta = meta
        self.gamma = gamma
        self.require_returns = require_returns

    def preproc_obsv(self, obsv: Any) -> Any:
        return obsv

    def preproc_action(self, action) -> Any:
        return self.meta.action_to_index(action)

    def preproc_info(self, info: Any) -> dict:
        return {}

    def postproc_obsvs(self, obsvs, episode_info) -> dict:
        return obsvs

    def postproc_trans(self, trans) -> dict:
        return trans

    def output_to_carrier(self, output):
        prev_v = output['v'].clone().detach()
        return output, {'prev_v': prev_v}

    @property
    def trans_keys(self):
        return None

    @property
    def obsv_keys(self):
        return None

    @property
    def next_obsv_keys(self):
        return []

    @property
    def full_obsv_keys(self):
        return []

    @property
    def future_obsv_keys(self):
        return []

    @property
    def next_full_obsv_keys(self):
        return []

    def combine_seq_info(self, episode_info: dict, obsv_info: dict, trans_info: dict, step: int, inference: bool, start_step: int = 0) \
            -> Tuple[dict, dict, dict, dict]:

        if inference:
            obsv_seq = get_dict(obsv_info, slice(start_step, step + 1), keys=self.obsv_keys)
            next_obsv_seq = {}
            curr_obsv_seq = get_dict(obsv_info, step)
            full_obsv_seq = get_dict(obsv_info, keys=self.full_obsv_keys)
            future_obsv_seq = {}
            next_full_obsv_seq = {}
            post_obsv_seq = {}
            curr_trans_seq = {}
            trans_seq = get_dict(trans_info, slice(start_step, step), keys=self.trans_keys)
            future_trans_seq = get_dict(trans_seq, slice(step, None))
            full_trans_seq = get_dict(trans_info, keys=self.trans_keys)
        else:
            obsv_seq = get_dict(obsv_info, slice(start_step, step + 1), keys=self.obsv_keys)
            next_obsv_seq = get_dict(obsv_info, slice(start_step + 1, step + 2), keys=self.next_obsv_keys)
            curr_obsv_seq = get_dict(obsv_info, step)
            post_obsv_seq = get_dict(obsv_info, step + 1)
            future_obsv_seq = get_dict(obsv_info, slice(step + 1, -1), keys=self.future_obsv_keys)
            full_obsv_seq = get_dict(obsv_info, slice(start_step, -1), keys=self.full_obsv_keys)
            next_full_obsv_seq = get_dict(obsv_info, slice(start_step + 1, None), keys=self.next_full_obsv_keys)
            curr_trans_seq = get_dict(trans_info, step, keys=self.trans_keys)
            trans_seq = get_dict(trans_info, slice(start_step, step + 1), keys=self.trans_keys)
            future_trans_seq = get_dict(trans_seq, slice(step + 1, None))
            full_trans_seq = get_dict(trans_info, slice(start_step, None), keys=self.trans_keys)

            if self.require_returns:
                trans_seq['returns'] = eval_returns(trans_seq['rewards'], gamma=self.gamma)

        return {**add_prefix(curr_obsv_seq, "curr_"), **add_prefix(curr_trans_seq, "curr_"), **add_prefix(post_obsv_seq, "post_")},\
               {**obsv_seq, **add_prefix(next_obsv_seq, "next_"), **trans_seq}, \
               {**add_prefix(future_obsv_seq, "future_"), **add_prefix(future_trans_seq, "future_")}, \
               {**add_prefix(full_obsv_seq, "_"), **add_prefix(next_full_obsv_seq, "_next_"), **add_prefix(full_trans_seq, "_")}

    def combine_info(self, curr_info, past_seq_info, future_seq_info, full_seq_info, step, inference: bool):
        return curr_info, past_seq_info, future_seq_info, full_seq_info

    @staticmethod
    def collate(batch: list):
        r"""concatenate batch elements with different sizes into one batch"""
        fixed_list = defaultdict(list)
        past_seq_list = defaultdict(list)
        future_seq_list = defaultdict(list)
        full_seq_list = defaultdict(list)
        for fixed, past, future, full in batch:
            for k, v in fixed.items():
                fixed_list[k].append(v)
            for k, v in past.items():
                past_seq_list[k].append(v)
            for k, v in future.items():
                future_seq_list[k].append(v)
            for k, v in full.items():
                full_seq_list[k].append(v)
        results = {}
        for k, v in fixed_list.items():
            try:
                results[k] = torch.stack(v) if isinstance(v[0], torch.Tensor) else torch.tensor(v)
            except: results[k] = v
        for l in [past_seq_list, future_seq_list, full_seq_list]:
            for k, v in l.items():
                try: results[k] = torch.cat(v)
                except: results[k] = v
        for l, p in zip([past_seq_list, future_seq_list, full_seq_list], ["", "future_", "_"]):
            if l:
                elem = next(iter(l.values()))
                results[f"{p}lens"] = torch.tensor([len(val) for i, val in enumerate(elem)], dtype=torch.long)
                results[f"{p}index"] = torch.cat([torch.full((len(val),), i, dtype=torch.long) for i, val in enumerate(elem)])
                results[f"{p}step"] = torch.cat([torch.arange(len(val), dtype=torch.long) for i, val in enumerate(elem)])
        results['batch_size'] = len(batch)
        return results

    @staticmethod
    def stats(info: dict):
        status, total_steps, opt_steps = [info[k] for k in ['status', 'total_steps', 'opt_steps']]
        result = {k.name: 0 for k in NavStatus}
        result[status.name] = 1
        result['acc'] = result[NavStatus.success.name]
        result['spl'] = opt_steps / max(total_steps, opt_steps) if status == NavStatus.success else 0
        return result

    def postproc_actions(self, actions):
        return [self.meta.action_from_index(a) for a in actions]

    def postproc_preds(self, preds):
        length = len(next(iter(preds.values())))
        result = []
        for i in range(length):
            data = {k: (v[i].cpu() if len(v) == length else v.cpu()) for k, v in preds.items() if k not in self.pred_exclude_keys and v is not None}
            result.append(data)
        return result

    @property
    def pred_exclude_keys(self):
        return ["d_logits", "p_logits"]


class NavHandler(DataHandler):
    @property
    def obsv_keys(self):
        return ['poses']

    @property
    def full_obsv_keys(self):
        return ['poses']

    @property
    def next_obsv_keys(self):
        return ['poses']

    @property
    def next_full_obsv_keys(self):
        return ['poses']

    @property
    def future_obsv_keys(self):
        return ['poses']


class VisNavHandler(NavHandler):
    def postproc_obsvs(self, obsvs, episode_info) -> dict:
        obsvs['rgb'] = obsvs['rgb'].float() / 255
        return obsvs

    @property
    def obsv_keys(self):
        return ['rgb', 'surf_xyz', 'poses']

    def output_to_carrier(self, output):
        carrier = {'prev_map_raw': output['feature_map_raw'].clone().detach(),
                   'prev_counts_raw': output['feature_counts_raw'].clone().detach(),
                   # 'prev_free_map_raw': output['free_map_raw'].clone().detach(),
                   'prev_rgb_map_raw': output['rgb_map_raw'].clone().detach(),
                   'prev_v': output['v'].clone().detach()}
        # del output['feature_map']
        return output, carrier
