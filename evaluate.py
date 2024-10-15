import os
import pandas as pd
from collections import Counter


eval_path = "evaluate_results"
eval_files = ""

# log_list = ['HDFS', 'Spark', 'BGL', 'Windows', 'Linux', 'Android', 'Mac', 'Hadoop', 'HealthApp', 'OpenSSH', 'Thunderbird', 'Proxifier', 'Apache', 'HPC', 'Zookeeper', 'OpenStack']
log_list = ['Android', 'Apache', 'BGL', 'Hadoop', 'HDFS', 'HealthApp', 'HPC', 'Linux',
            'Mac', 'OpenSSH', 'OpenStack', 'Proxifier', 'Spark', 'Thunderbird', 'Windows', 'Zookeeper']


def generateDifferent(groundtruth, result):
    # len(predicted_list) may smaller than len(groundtruth)
    length = len(result['template'])
    if length == 0:
        return 0
    correct = 0
    df_diff = pd.DataFrame(columns=['log', 'template_parsed', 'gt'])
    for i in range(length):
        if result['template'][i] == groundtruth.loc[groundtruth['Content'] == result['log'][i]]['EventTemplate'].values[0]:
            correct += 1
        else:
            df_diff.loc[len(df_diff)] = [result['log'][i], result['template'][i], groundtruth.loc[groundtruth['Content'] == result['log'][i]]['EventTemplate'].values[0]]
    return df_diff
def evaluatePA(groundtruth, result):
    # len(predicted_list) may smaller than len(groundtruth)
    length = len(result['template'])
    if length == 0: return 0
    correct = 0
    for i in range(length):
        if result['template'][i] == groundtruth.loc[groundtruth['Content'] == result['log'][i]]['EventTemplate'].values[0]:
            correct += 1
    return correct / length


# correctly identified templates over total num of identified template
def evaluatePTA(groundtruth, result):
    # generate a "template: log indexes list" mapping for groundtruth
    oracle_tem_dict = {}
    for idx in range(len(result['template'])):
        if groundtruth['EventTemplate'][idx] not in oracle_tem_dict:
            oracle_tem_dict[groundtruth['EventTemplate'][idx]] = [groundtruth['Content'][idx]]
        else:
            oracle_tem_dict[groundtruth['EventTemplate'][idx]].append(groundtruth['Content'][idx])

    # generate mapping for identified template
    result_tem_dict = {}
    for idx in range(len(result['template'])):
        if result['template'][idx] not in result_tem_dict:
            result_tem_dict[result['template'][idx]] = [result['log'][idx]]
        else:
            result_tem_dict[result['template'][idx]].append(result['log'][idx])

    correct_num = 0
    for key in result_tem_dict.keys():
        if key not in oracle_tem_dict:
            continue
        else:
            if Counter(oracle_tem_dict[key]) == Counter(result_tem_dict[key]): correct_num += 1

    return correct_num / len(result_tem_dict)


# correctly identified templates over total num of oracle template
def evaluateRTA(groundtruth, result):
    # generate a "template: log indexes list" mapping for groundtruth
    oracle_tem_dict = {}
    for idx in range(len(result['template'])):
        if groundtruth['EventTemplate'][idx] not in oracle_tem_dict:
            oracle_tem_dict[groundtruth['EventTemplate'][idx]] = [groundtruth['Content'][idx]]
        else:
            oracle_tem_dict[groundtruth['EventTemplate'][idx]].append(groundtruth['Content'][idx])

    # generate mapping for identified template
    result_tem_dict = {}
    for idx in range(len(result['template'])):
        if result['template'][idx] not in result_tem_dict:
            result_tem_dict[result['template'][idx]] = [result['log'][idx]]
        else:
            result_tem_dict[result['template'][idx]].append(result['log'][idx])

    correct_num = 0
    for key in oracle_tem_dict.keys():
        if key not in result_tem_dict:
            continue
        else:
            if Counter(oracle_tem_dict[key]) == Counter(result_tem_dict[key]): correct_num += 1

    return correct_num / len(oracle_tem_dict)


# calculate grouping accuracy
def evaluateGA(groundtruth, result):
    # load logs and templates
    compared_list = result['log'].tolist()

    # select groundtruth logs that have been parsed
    parsed_idx = []
    for idx, row in groundtruth.iterrows():
        if row['Content'] in compared_list:
            parsed_idx.append(idx)
            compared_list.remove(row['Content'])

    if not (len(parsed_idx) == 2000):
        print(len(parsed_idx))
        print("Wrong number of groundtruth logs!")
        return 0

    groundtruth = groundtruth.loc[parsed_idx]

    # grouping
    groundtruth_dict = {}
    for idx, row in groundtruth.iterrows():
        if row['EventTemplate'] not in groundtruth_dict:
            # create a new key
            groundtruth_dict[row['EventTemplate']] = [row['Content']]
        else:
            # add the log in an existing group
            groundtruth_dict[row['EventTemplate']].append(row['Content'])

    result_dict = {}
    for idx, row in result.iterrows():
        if row['template'] not in result_dict:
            # create a new key
            result_dict[row['template']] = [row['log']]
        else:
            # add the log in an existing group
            result_dict[row['template']].append(row['log'])

    # sorting for comparison
    for key in groundtruth_dict.keys():
        groundtruth_dict[key].sort()

    for key in result_dict.keys():
        result_dict[key].sort()

    # calculate grouping accuracy
    count = 0
    for parsed_group_list in result_dict.values():
        for gt_group_list in groundtruth_dict.values():
            if parsed_group_list == gt_group_list:
                count += len(parsed_group_list)
                break

    return count / 2000

if not os.path.exists(eval_path + "/evaluate" +eval_files + ".csv"):
    df = pd.DataFrame(columns=['Dataset', 'Parsing Accuracy', 'Precision Template Accuracy', 'Recall Template Accuracy', 'Grouping Accuracy'])
else:
    df = pd.read_csv(eval_path + "/evaluate" + eval_files + ".csv")

for log_name in log_list:
    log_path = "data/loghub_2k/" + log_name + "/" + log_name + "_2k.log_structured.csv"
    result_path = eval_files + "/" + "2000_" + log_name + "_result.csv"
    df_groundtruth = pd.read_csv(log_path)
    df_parsedlog = pd.read_csv(result_path)
    df_diff = generateDifferent(df_groundtruth, df_parsedlog)
    PA = evaluatePA(df_groundtruth, df_parsedlog)
    PTA = evaluatePTA(df_groundtruth, df_parsedlog)
    RTA = evaluateRTA(df_groundtruth, df_parsedlog)
    GA = evaluateGA(df_groundtruth, df_parsedlog)
    print("Evalute " + eval_files + " of dataset " + log_name +  " results:")
    print("{}:\t PA:\t{:.6f}\tPTA:\t{:.6f}\tRTA:\t{:.6f}\tGA:\t{:.6f}".format(log_name, PA, PTA, RTA, GA))
    if log_name not in df['Dataset'].values:
        df.loc[len(df)] = [log_name, PA, PTA, RTA, GA]
    else:
        df.loc[df['Dataset'] == log_name, 'Parsing Accuracy'] = PA
        df.loc[df['Dataset'] == log_name, 'Precision Template Accuracy'] = PTA
        df.loc[df['Dataset'] == log_name, 'Recall Template Accuracy'] = RTA
        df.loc[df['Dataset'] == log_name, 'Grouping Accuracy'] = GA
    df.to_csv(eval_path + "/evaluate" + eval_files + ".csv", index=False, float_format="%.6f")
    df_diff.to_csv(eval_path + "/evaluate" + eval_files + "_" + log_name + "_different_logs.csv", index=False)