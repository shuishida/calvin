from typing import Tuple
from core.handler import NavHandler


class GridDataHandler(NavHandler):
    def combine_info(self, curr_info, past_seq_info, future_seq_info, full_seq_info, step, inference: bool) -> Tuple[dict, dict, dict, dict]:
        feature_map = curr_info['curr_feature_map']
        return {**curr_info, 'feature_map': feature_map}, past_seq_info, future_seq_info, full_seq_info
