from run_a2c import a2c
from util import *
from datetime import datetime

dataType = "real"
dataId = "vic"
alpha = 100
beta = 1

usersNum_list = [2000]
facsNum_list = [40]
sitesNum_list = [40]

input_rawdata_file = ""
input_rawdata_user_file = ""
input_rawdata_fac_file = ""
output_result_path = "result.txt"

for usersNum_t in usersNum_list:
    for facsNum_t in facsNum_list:
        for sitesNum_t in sitesNum_list:
            usersNum = usersNum_t
            facsNum = facsNum_t
            sitesNum = sitesNum_t

            if dataType == "synthetic":
                input_rawdata_file = "dataset/" + dataType + "/dataset_TSMC2014_NYC.txt"

            elif dataType == "real":
                input_rawdata_user_file = "dataset/" + dataType + "/user.csv"
                input_rawdata_fac_file = "dataset/" + dataType + "/facility.csv"

            output_path = "dataset/" + dataType + "/" + dataId + "_" + str(usersNum) + "_" + str(
                facsNum) + "_" + str(sitesNum) + ".txt"

            input_path = output_path

            user_fac_dis_map, user_fac_one_old_map, user_dis_one_old_map = readData(input_path, usersNum, facsNum)
            dis_max_threshold = getMaxDis(user_fac_dis_map)

            startTime = datetime.now()
            obj, fac_state_list = a2c(usersNum, facsNum, sitesNum, user_fac_dis_map, user_dis_one_old_map,
                                      dis_max_threshold, alpha, beta)

            print("The objective is: " + str(obj))
            endTime = datetime.now()
            duration = endTime - startTime
            print("The runtime is: " + str(duration.total_seconds()))

            obj, obj1, obj2, obj3 = computeObj(usersNum, user_fac_dis_map, user_dis_one_old_map, dis_max_threshold,
                                               fac_state_list,
                                               alpha, beta)

            with open(output_result_path, 'a') as fw:
                fw.write(str(obj) + "\t" + str(obj1) + "\t" + str(obj2) + "\t" + str(
                    obj3) + "\t" + str(duration.total_seconds()) + "\t" + str(
                    usersNum) + "\t" + str(facsNum) + "\t" + str(sitesNum) + "\t" + str(alpha) + "\t" + str(
                    beta) + "\t" + dataType + "\t" + str(fac_state_list) + "\n")
            fw.close()
