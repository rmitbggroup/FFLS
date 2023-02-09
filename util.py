import pickle, sys
import numpy as np
import os
import errno
import math


def computeDis(lat1, lat2, lng1, lng2):
    num1 = pow(lat1 - lat2, 2)
    num2 = pow(lng1 - lng2, 2)
    num = num1 + num2
    return math.sqrt(num)


def isfloat_str(str_number):
    try:
        float(str_number)
        return True
    except ValueError:
        return False


def getMaxDis(user_fac_dis_map):
    dis_max_threshold = -1
    for userId in user_fac_dis_map:
        for facId in user_fac_dis_map[userId]:
            if user_fac_dis_map[userId][facId] > dis_max_threshold:
                dis_max_threshold = user_fac_dis_map[userId][facId]
    return dis_max_threshold


def readData(input_path, usersNum, facsNum):
    user_fac_dis_map = {}
    with open(input_path, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line_list = line.strip().split("\t")
            userId = int(line_list[0])
            facId = int(line_list[1])
            dis = float(line_list[2])
            if userId not in user_fac_dis_map:
                fac_dis_map = {}
                fac_dis_map[facId] = dis
                user_fac_dis_map[userId] = fac_dis_map
            else:
                user_fac_dis_map[userId][facId] = dis

    user_fac_one_old_map = {}
    user_dis_one_old_map = {}

    for i in range(usersNum):  # i: userId
        for j in user_fac_dis_map[i]:  # i: facId
            # user_fac_dis_one_old_map[i] =
            if j >= facsNum:
                continue
            else:
                user_fac_one_old_map[i] = j
                user_dis_one_old_map[i] = user_fac_dis_map[i][j]
                break
    return user_fac_dis_map, user_fac_one_old_map, user_dis_one_old_map


def readPklData():
    input_file = "logdispatch_simulator/experiments/20210827_03-00/results.pkl"
    objects = []
    with (open(input_file, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    print(objects)


def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def computeObj(usersNum, user_fac_dis_map, user_dis_one_old_map, dis_max_threshold, fac_state_cmp, alpha, beta):
    obj1 = 0
    facsNum = len(fac_state_cmp)

    fac_user_list_t = {}
    user_dis_one_new_map = {}
    for i in fac_state_cmp:
        fac_user_list_t[i] = 0

    for userId in range(usersNum):
        for facId in user_fac_dis_map[userId]:
            if facId in fac_state_cmp:
                fac_user_list_t[facId] += 1
                user_dis_one_new_map[userId] = user_fac_dis_map[userId][facId]
                break

    fac_expo_avg = usersNum / facsNum
    for i in fac_state_cmp:
        obj1 += math.pow(fac_expo_avg - fac_user_list_t[i], 2)
    obj1 = float(obj1) / usersNum / facsNum

    obj2 = 0

    for i in range(usersNum):
        obj2 += user_dis_one_new_map[i]
    obj2 = float(obj2) / dis_max_threshold

    obj = alpha * obj1 + beta * obj2
    return obj, alpha * obj1, beta * obj2, -1


def computeFacExpoAndUserConv(usersNum, user_fac_dis_map, dis_max_threshold, fac_state_cmp, alpha, beta):
    obj1 = 0
    facsNum = len(fac_state_cmp)

    fac_user_list_t = {}
    user_dis_one_new_map = {}
    for i in fac_state_cmp:
        fac_user_list_t[i] = 0

    for userId in range(usersNum):
        for facId in user_fac_dis_map[userId]:
            if facId in fac_state_cmp:
                fac_user_list_t[facId] += 1
                user_dis_one_new_map[userId] = user_fac_dis_map[userId][facId]
                break
    fac_expo_avg = usersNum / facsNum
    for i in fac_state_cmp:
        obj1 += math.pow(fac_expo_avg - fac_user_list_t[i], 2)
    obj1 = float(obj1) / usersNum / facsNum

    obj2 = 0
    for i in range(usersNum):
        obj2 += user_dis_one_new_map[i]
    obj2 = float(obj2) / dis_max_threshold

    obj = alpha * obj1 + beta * obj2
    return obj, alpha * obj1, beta * obj2, fac_user_list_t, user_dis_one_new_map


def compute_result_stats(usersNum, user_fac_dis_map, fac_state_cmp):
    fac_user_list_t = {}
    user_dis_one_new_map = {}
    for i in fac_state_cmp:
        fac_user_list_t[i] = []

    for userId in range(usersNum):
        for facId in user_fac_dis_map[userId]:
            if facId in fac_state_cmp:
                fac_user_list_t[facId].append(userId)
                user_dis_one_new_map[userId] = 111 * user_fac_dis_map[userId][facId]
                break

    dis_list = list(user_dis_one_new_map.values())

    fac_exp_list = []
    for i in fac_state_cmp:
        exp_num = len(fac_user_list_t[i])
        fac_exp_list.append(exp_num)

    dis_mean = get_list_mean(dis_list)
    dis_std = get_list_std(dis_list)
    exp_mean = get_list_mean(fac_exp_list)
    exp_std = get_list_std(fac_exp_list)
    return dis_mean, dis_std, exp_mean, exp_std


def get_list_mean(data_list):
    return np.mean(data_list)


def get_list_std(data_list):
    return np.std(data_list)
