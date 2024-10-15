import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import math
from collections import Counter
import json
import os
from sklearn.cluster import DBSCAN

# BM25 parameters
k1 = 1.5
b = 0.75

N = 5

def compute_idf(corpus):
    # 计算IDF
    idf = {}
    N = len(corpus)
    for doc in corpus:
        for word in set(doc):
            idf[word] = idf.get(word, 0) + 1

    for word, freq in idf.items():
        idf[word] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

    return idf


def compute_bm25(doc, query, idf, avg_doc_len):
    score = 0.0
    doc_len = len(doc)
    doc_counter = Counter(doc)

    for word in query:
        if word in idf:
            tf = doc_counter[word]
            numerator = idf[word] * tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
            score += numerator / denominator

    return score


def bm25_similarity(log, sample_contents, idf, avg_doc_len):
    log_tokens = log.split()
    scores = []

    for sample_log in sample_contents:
        sample_tokens = sample_log.split()
        score = compute_bm25(sample_tokens, log_tokens, idf, avg_doc_len)
        scores.append(score)

    return np.array(scores)


def find_top_N_similar_logs(current_log, sample_df, N):
    sample_contents = sample_df['Content'].values
    sample_templates = sample_df['EventTemplate'].values

    # 计算IDF和平均文档长度
    sample_tokens_list = [log.split() for log in sample_contents]
    avg_doc_len = np.mean([len(tokens) for tokens in sample_tokens_list])
    idf = compute_idf(sample_tokens_list)

    list_contents = []
    list_templates = []

    scores = bm25_similarity(current_log, sample_contents, idf, avg_doc_len)
    # top_N_indices = np.argsort(scores)[-N:][::-1]  # 从大到小排序 默认降序
    #从小到大
    top_N_indices = np.argsort(scores)[-N:]

    # print(scores[top_N_indices])

    top_N_logs = sample_contents[top_N_indices]
    top_N_templates = sample_templates[top_N_indices]
    for log, template in zip(top_N_logs, top_N_templates):
        list_contents.append(log)
        list_templates.append(template)

    return list_contents, list_templates

# 加权聚类方法
def sampleLogs(dataset, sample_ratio):
    # 读取CSV文件
    df = pd.read_csv('../data/loghub_2k/' + dataset + '/' + dataset + '_2k.log_structured.csv')

    # 假设目标列名为'log_text'
    log_texts = df['Content'].values

    # 计算每条日志的复杂性（如token数目和字符串长度）
    token_counts = [len(log.split()) for log in log_texts]
    lengths = [len(log) for log in log_texts]

    # 计算复杂性得分
    complexity_scores = [a**a + b for a, b in zip(token_counts, lengths)]
    complexity_scores_np = np.array(complexity_scores).reshape(-1, 1)

    # 将复杂性得分添加到DataFrame中
    df['token_count'] = token_counts
    df['length'] = lengths
    df['complexity_score'] = complexity_scores

    # 使用DBSCAN进行聚类
    # 选择合适的 eps（距离参数）和 min_samples（每个簇的最小样本数）
    dbscan = DBSCAN(eps=1.9, min_samples=1, metric='euclidean')
    df['cluster'] = dbscan.fit_predict(complexity_scores_np.reshape(-1, 1))

    # DBSCAN 中 -1 表示噪声点，不属于任何簇
    num_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)
    print(f"Number of clusters found by DBSCAN: {num_clusters}")

    # 在每个簇内进行加权采样
    selected_logs = []

    for cluster in range(num_clusters):
        cluster_data = df[df['cluster'] == cluster]

        if (len(cluster_data) == 0):
            continue
        # 确定簇内需要的样本数（向上取整）
        num_samples_per_cluster = math.ceil(len(cluster_data) * sample_ratio)

        # 确保簇内样本数不小于需要的样本数
        if len(cluster_data) < num_samples_per_cluster:
            num_samples_per_cluster = len(cluster_data)

        # 添加平滑项，增加低复杂性样本的权重
        min_complexity = cluster_data['complexity_score'].min()
        max_complexity = cluster_data['complexity_score'].max()
        smoothing_factor = (max_complexity - min_complexity) * 0.2  # 调整平滑因子的大小

        # 计算加权概率
        weights = (cluster_data['complexity_score'] + smoothing_factor) / (
                    cluster_data['complexity_score'] + smoothing_factor).sum()
        # 确保权重之和为1
        weights = weights / weights.sum()

        selected_indices = np.random.choice(cluster_data.index, size=num_samples_per_cluster, replace=False, p=weights)

        selected_logs.extend(cluster_data.loc[selected_indices, 'Content'].values)

    corresponding_templates = []
    for log in selected_logs:
        temp = df.loc[df['Content'] == log, 'EventTemplate'].values[0]
        corresponding_templates.append(temp)
    result_df = pd.DataFrame({
        'Content': selected_logs,
        'EventTemplate': corresponding_templates
    })
    # print(result_df)
    return result_df


def simple_random_sampling(df, sample_ratio):

    log_texts = df['Content'].values

    # 计算需要采样的数量
    num_samples = int(len(log_texts) * sample_ratio)

    # 随机选择num_samples个样本
    selected_indices = np.random.choice(len(log_texts), num_samples, replace=False)

    # 根据随机选择的索引获取对应的日志内容
    selected_logs = df.loc[selected_indices, 'Content'].values

    # 从原数据集中删除选中的日志，构建测试集
    remaining_df = df.drop(selected_indices)

    # 保存测试集到CSV文件，放在与原文件相同的目录下
    test_file_path = '../data/loghub_2k/' + dataset + '/' + dataset + '_2k.log_structured_random_test.csv'
    remaining_df.to_csv(test_file_path, index=False)

    return selected_logs


def sample_train_contents(df, sample_ratio):
    log_texts = df['Content'].values

    # 计算每条日志的复杂性（如token数目和字符串长度）
    token_counts = [len(log.split()) for log in log_texts]
    lengths = [len(log) for log in log_texts]

    # 计算复杂性得分
    complexity_scores = [a ** a + b for a, b in zip(token_counts, lengths)]
    complexity_scores_np = np.array(complexity_scores).reshape(-1, 1)

    # 将复杂性得分添加到DataFrame中
    df['token_count'] = token_counts
    df['length'] = lengths
    df['complexity_score'] = complexity_scores

    # 使用DBSCAN进行聚类
    # 选择合适的 eps（距离参数）和 min_samples（每个簇的最小样本数）
    dbscan = DBSCAN(eps=1.9, min_samples=1, metric='euclidean')
    df['cluster'] = dbscan.fit_predict(complexity_scores_np.reshape(-1, 1))

    # DBSCAN 中 -1 表示噪声点，不属于任何簇
    num_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)
    print(f"Number of clusters found by DBSCAN: {num_clusters}")

    # 在每个簇内进行加权采样
    selected_logs = []
    selected_indices = []

    for cluster in range(num_clusters):
        cluster_data = df[df['cluster'] == cluster]

        if (len(cluster_data) == 0):
            continue
        # 确定簇内需要的样本数（向上取整）
        num_samples_per_cluster = math.ceil(len(cluster_data) * sample_ratio)

        # 确保簇内样本数不小于需要的样本数
        if len(cluster_data) < num_samples_per_cluster:
            num_samples_per_cluster = len(cluster_data)

        # 添加平滑项，增加低复杂性样本的权重
        min_complexity = cluster_data['complexity_score'].min()
        max_complexity = cluster_data['complexity_score'].max()
        smoothing_factor = (max_complexity - min_complexity) * 0.2  # 调整平滑因子的大小

        # 计算加权概率
        weights = (cluster_data['complexity_score'] + smoothing_factor) / (
                cluster_data['complexity_score'] + smoothing_factor).sum()
        # 确保权重之和为1
        weights = weights / weights.sum()

        # 选择样本
        cluster_selected_indices = np.random.choice(cluster_data.index, size=num_samples_per_cluster, replace=False,
                                                    p=weights)
        selected_logs.extend(cluster_data.loc[cluster_selected_indices, 'Content'].values)
        selected_indices.extend(cluster_selected_indices)

    # 构建测试集，即未被选中的日志
    remaining_df = df.drop(selected_indices)

    # 保存测试集到CSV文件
    test_file_path = '../data/loghub_2k/' + dataset + '/' + dataset + '_2k.log_structured_DBSCAN_test.csv'
    remaining_df.to_csv(test_file_path, index=False)

    return selected_logs

if __name__ == '__main__':
    datasets = ['HDFS', 'Spark', 'BGL', 'Windows', 'Linux', 'Android', 'Mac', 'Hadoop', 'HealthApp', 'OpenSSH',
                'Thunderbird', 'Proxifier', 'Apache', 'HPC', 'Zookeeper', 'OpenStack']
    # datasets = ['Proxifier']
    instruction = "For each log after <content> tag, try your best to extract one log template\
            (substitute variable tokens in the log as <*> and remain constant tokens to construct the template)\
            and put the template after <template> tag and between <START> and <END> tags."
    train_sample_ratio = 0.15
    for dataset in datasets:
        data = []
        # for sample_ratio in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        for sample_ratio in [0.15]:
            sample_df = sampleLogs(dataset, sample_ratio)
            df = pd.read_csv('data/loghub_2k/' + dataset + '/' + dataset + '_2k.log_structured.csv')
            contents = df['Content'].tolist()
            # train_contents = sample_train_contents(df, train_sample_ratio)
            train_contents = simple_random_sampling(df, train_sample_ratio)
            for log in train_contents:
                gt = df.loc[df['Content'] == log, 'EventTemplate'].values[0]
                output = "<START> " + gt + " <END>\n"
                similar_contents, similar_templates = find_top_N_similar_logs(log, sample_df, N)
                for i in range(N+1): # 0-shot to N-shot
                    prompt = "\n"
                    for j in range(i-1, -1, -1):
                        prompt = (prompt + "<content>: " + similar_contents[j] + "\n"
                                  + "<template>: <START> " + similar_templates[j] + " <END>\n")
                    prompt = prompt + "<content>: " + log + "\n<template>: "
                    # data of MicJson
                    current_data = {
                        "instruction": instruction,
                        "input": prompt,
                        "output": output
                    }
                    data.append(current_data)

            print(len(data))
            # dir = f"Mic_tuning_data_DBSCAN_{train_sample_ratio}_for_train"
            dir = f"Mic_tuning_data_randomselect_{train_sample_ratio}_for_train"
            if not os.path.exists(dir):
                os.mkdir(dir)
            directory = dir + '/Sample_ratio_' + str(sample_ratio)
            if not os.path.exists(directory):
                os.mkdir(directory)
            print(f"当前工作目录：{os.getcwd()}")

            current_dataset_file = dataset + f'_meta_tuning_data_sample_{train_sample_ratio}.json'
            file_path = os.path.join(directory, current_dataset_file)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            print(f"JSON文件已生成：{file_path}")




