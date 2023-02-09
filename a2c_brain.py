import logging
import numpy as np

from util import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger_ch = logging.StreamHandler()
logger_ch.setLevel(logging.DEBUG)
logger_ch.setFormatter(logging.Formatter(
    '%(asctime)s[%(levelname)s][%(lineno)s:%(funcName)s]||%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(logger_ch)
RANDOM_SEED = 0  # unit test use this random seed.


class UserFacRelation:
    '''M: usersNum; N: facsNum; A user facility relation consists of M*N elements '''

    def __init__(self, usersNum, facsNum, sitesNum, user_fac_dis_map, user_dis_one_old_map, dis_max_threshold, alpha, beta):

        self.usersNum = usersNum
        self.facsNum = facsNum
        self.sitesNum = sitesNum
        action_dim = sitesNum + 1  # move or not
        state_dim = 1 + sitesNum

        action_short_dim = 2  # move or not: 0 (keep still); 1 (move)
        state_short_dim = 1 + sitesNum  # 0: the current pos; >0: move to the i-1 site
        self.action_dim = action_dim
        self.action_short_dim = action_short_dim
        self.state_dim = state_dim
        self.state_short_dim = state_short_dim
        self.RANDOM_SEED = RANDOM_SEED
        self.user_fac_dis_map = user_fac_dis_map
        self.user_dis_one_old_map = user_dis_one_old_map
        self.dis_max_threshold = dis_max_threshold
        self.alpha = alpha
        self.beta = beta

    def reset_randomseed(self, random_seed):
        self.RANDOM_SEED = random_seed

    def reset_clean(self):
        initial_state = np.zeros((self.facsNum, self.state_dim))
        fac_pos_map = {}
        for i in range(self.facsNum):
            initial_state[i, 0] = 1
            fac_pos_map[i] = i

        initial_goal, _, _, _ = computeObj(self.usersNum, self.user_fac_dis_map, self.user_dis_one_old_map, self.dis_max_threshold, list(fac_pos_map.values()), self.alpha, self.beta)
        return initial_state, fac_pos_map, initial_goal

    def reset_clean_short(self):
        # the first col means keeping still
        initial_state = np.zeros((self.facsNum, self.state_short_dim))
        fac_pos_map = {}
        for i in range(self.facsNum):
            initial_state[i, 0] = 1
            fac_pos_map[i] = i

        initial_goal, _, _, _ = computeObj(self.usersNum, self.user_fac_dis_map, self.user_dis_one_old_map, self.dis_max_threshold, list(fac_pos_map.values()), self.alpha, self.beta)
        return initial_state, fac_pos_map, initial_goal

    def step(self, dispatch_actions, curr_state, fac_pos_map):
        next_state = curr_state
        for action in dispatch_actions:
            start_node_id, end_node_id = action
            flag = True
            if end_node_id > 0:
                if np.sum(curr_state[:, end_node_id]) > 1:
                    flag = False

            if flag == True:
                next_state[start_node_id] = 0
                next_state[start_node_id, end_node_id] = 1
                if end_node_id == 0:
                    fac_pos_map[start_node_id] = start_node_id
                else:
                    fac_pos_map[start_node_id] = end_node_id - 1 + self.facsNum
        curr_goal, _, _, _ = computeObj(self.usersNum, self.user_fac_dis_map, self.user_dis_one_old_map, self.dis_max_threshold, list(fac_pos_map.values()), self.alpha, self.beta)

        return next_state, curr_goal, fac_pos_map

    def step_short(self, dispatch_actions, curr_state, fac_pos_map):
        next_state = curr_state
        for action in dispatch_actions:
            start_node_id, curr_action_id = action
            if curr_action_id == 0:
                fac_pos_map[start_node_id] = start_node_id
                curr_goal, _, _, _ = computeObj(self.usersNum, self.user_fac_dis_map, self.user_dis_one_old_map, self.dis_max_threshold, list(fac_pos_map.values()), self.alpha, self.beta)
                next_state[start_node_id] = 0
                next_state[start_node_id, 0] = 1
            elif curr_action_id == 1:
                min_goal = 100000
                min_end_node_id = -1
                occupied_site_list = list(fac_pos_map.values())
                for i in range(self.facsNum, self.facsNum + self.sitesNum):
                    temp_fac_pos_map = fac_pos_map
                    if i not in occupied_site_list:
                        temp_fac_pos_map[start_node_id] = i
                        temp_goal, _, _, _ = computeObj(self.usersNum, self.user_fac_dis_map, self.user_dis_one_old_map, self.dis_max_threshold, list(temp_fac_pos_map.values()), self.alpha, self.beta)
                        if temp_goal < min_goal:
                            min_end_node_id = i
                            min_goal = temp_goal

                temp_fac_pos_map = fac_pos_map
                temp_fac_pos_map[start_node_id] = start_node_id
                temp_goal, _, _, _ = computeObj(self.usersNum, self.user_fac_dis_map, self.user_dis_one_old_map, self.dis_max_threshold, list(temp_fac_pos_map.values()), self.alpha, self.beta)
                if temp_goal < min_goal:
                    min_end_node_id = start_node_id
                    min_goal = temp_goal

                next_state[start_node_id] = 0
                end_node_id = min_end_node_id
                fac_pos_map[start_node_id] = end_node_id
                curr_goal = min_goal
                if end_node_id != start_node_id:
                    next_state[
                        start_node_id, end_node_id - self.facsNum + 1] = 1  # plus 1 is because the first col means keeping still
                else:
                    next_state[start_node_id, 0] = 1
        return next_state, curr_goal, fac_pos_map

    def normalize_state(self, curr_state):
        max_value = np.max(curr_state)
        min_value = np.max(curr_state)
        if max_value == min_value:
            curr_state = np.zeros_like(curr_state)
        else:
            curr_state = (curr_state-min_value) / (max_value - min_value)
        return curr_state