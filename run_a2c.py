import pickle, sys
import time

sys.path.append("../")

from a2c_env import *
from a2c_brain import *
from util import *


def a2c(usersNum, facsNum, sitesNum, user_fac_dis_map, user_dis_one_old_map, dis_max_threshold, alpha, beta):
    dir_prefix = "log"
    current_time = time.strftime("%Y%m%d_%H-%M_%S")
    if trainedFlag == 1:
        log_dir = dir_prefix + "dispatch_simulator/experiments/{}/".format(current_time)
        mkdir_p(log_dir)
    else:
        log_dir = dir_prefix + "dispatch_simulator/experiments/{}/20210912_18-24_29"
    print("log dir is {}".format(log_dir))
    print("finish load data")

    env = UserFacRelation(usersNum, facsNum, sitesNum, user_fac_dis_map, user_dis_one_old_map, dis_max_threshold, alpha,
                          beta)

    action_short_dim = env.action_short_dim
    state_short_dim = env.state_short_dim

    episode_goals = []
    episode_fac_pos_maps = []

    startTime = datetime.now()
    tf.reset_default_graph()
    sess = tf.Session()
    tf.set_random_seed(1)
    q_estimator = Estimator(sess, action_short_dim,
                            state_short_dim,
                            env,
                            scope="q_estimator",
                            summaries_dir=log_dir)

    sess.run(tf.global_variables_initializer())

    replay = ReplayMemory(memory_size=1e+6, batch_size=int(3e+3))
    policy_replay = policyReplayMemory(memory_size=1e+6, batch_size=int(3e+3))
    stateprocessor = stateProcessor(action_short_dim, facsNum)

    # restore = True
    saver = tf.train.Saver()

    global_step1 = 0
    global_step2 = 0
    curr_state, fac_pos_map, pre_goal = env.reset_clean_short()
    curr_state = env.normalize_state(curr_state)
    action_tuple_pre = []
    for n_iter in np.arange(max_iter):
        action_tuple, valid_action_prob_mat, policy_state, action_choosen_mat, \
        curr_state_value = q_estimator.action(curr_state)
        if action_tuple_pre != action_tuple:
            next_state, curr_goal, fac_pos_map = env.step_short(action_tuple, curr_state, fac_pos_map)
        else:
            next_state = curr_state
            curr_goal = pre_goal

        next_state = env.normalize_state(next_state)
        reward = pre_goal - curr_goal

        if n_iter != 0 and trainedFlag == 1:
            r_grid = reward
            targets_batch = q_estimator.compute_targets(action_mat_prev, next_state, r_grid, gamma)

            advantage = q_estimator.compute_advantage(curr_state_value_prev, next_state, r_grid, gamma)

            action_mat = stateprocessor.to_action_short_mat(action_tuple)
            replay.add(curr_state, action_mat, targets_batch, next_state)

            policy_replay.add(policy_state_prev, action_choosen_mat_prev, advantage)

        action_tuple_pre = action_tuple
        pre_goal = curr_goal
        curr_state = next_state

        action_mat_prev = valid_action_prob_mat

        action_choosen_mat_prev = action_choosen_mat
        policy_state_prev = policy_state
        curr_state_value_prev = curr_state_value

        global_step1 += 1
        global_step2 += 1

        episode_goals.append(curr_goal)
        episode_fac_pos_maps.append(list(fac_pos_map.values()))

        print("iteration {} ********* reward {} ********* curr_goal {}".format(n_iter, reward, curr_goal))

        final_goal = np.min(episode_goals)
        final_goal_idx = np.argmin(episode_goals)
        final_fac_pos_list = episode_fac_pos_maps[final_goal_idx]

        pickle.dump([1], open(log_dir + "results.pkl", "wb"))

        if n_iter == 10000:
            break

        if n_iter != 0:
            for _ in np.arange(2):
                batch_s, _, batch_r, _ = replay.sample()
                iloss = q_estimator.update_value(batch_s, batch_r, 1e-3, global_step1)
                global_step1 += 1

            for _ in np.arange(2):
                batch_s, batch_a, batch_r = policy_replay.sample()
                q_estimator.update_policy(batch_s, batch_r.reshape([-1, 1]), batch_a, learning_rate,
                                          global_step2)
                global_step2 += 1

        if n_iter % 4 == 0:
            endTime = datetime.now()
            duration = endTime - startTime
            print("The runtime is: " + str(duration.total_seconds()))
            final_goal_idx = np.argmin(episode_goals)
            final_fac_pos_list = episode_fac_pos_maps[final_goal_idx]

            fac_state_list = final_fac_pos_list

            dis_mean, dis_std, exp_mean, exp_std = compute_result_stats(usersNum, user_fac_dis_map, fac_state_list)

            obj, obj1, obj2, obj3 = computeObj(usersNum, user_fac_dis_map, user_dis_one_old_map, dis_max_threshold,
                                               fac_state_list,
                                               alpha, beta)

            output_result_path = "result.txt"
            with open(output_result_path, 'a') as fw:
                fw.write(str(3) + "\t" + str(obj) + "\t" + str(obj1) + "\t" + str(obj2) + "\t" + str(
                    obj3) + "\t" + str(dis_mean) + "\t" + str(dis_std) + "\t" + str(
                    exp_mean) + "\t" + str(exp_std) + "\t" + str(duration.total_seconds()) + "\t" + str(
                    usersNum) + "\t" + str(facsNum) + "\t" + str(sitesNum) + "\t" + str(alpha) + "\t" + str(
                    beta) + "\t" + "real" + "\t" + str(fac_state_list) + "\n")
            fw.close()

    saver.save(sess, log_dir + "model.ckpt")
    endTime = datetime.now()
    duration = endTime - startTime
    print("The runtime is: " + str(duration.total_seconds()))
    final_goal = np.min(episode_goals)
    final_goal_idx = np.argmin(episode_goals)
    final_fac_pos_list = episode_fac_pos_maps[final_goal_idx]
    return final_goal, final_fac_pos_list
