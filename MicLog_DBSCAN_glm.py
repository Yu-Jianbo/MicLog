import os
import numpy as np
import pandas as pd
import re
import time
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


def validate_log_transformation(log1, log2):
    # 分割 log2，按 <*> 为界
    log2_parts = log2.split('<*>')

    # 初始化当前位置
    start_idx = 0
    errors = []
    replacements = []

    for part in log2_parts:
        if part.strip() == "":  # 忽略空的部分
            continue
        idx = log1.find(part.strip(), start_idx)
        if idx == -1:
            errors.append(part.strip())
        else:
            # 记录 <*>
            if start_idx != idx:
                replacements.append(log1[start_idx:idx].strip())
            start_idx = idx + len(part.strip())

    # 检查结尾是否有剩余部分未匹配
    if start_idx < len(log1):
        replacements.append(log1[start_idx:].strip())

    if errors:
        return False, errors, replacements
    return True, [], replacements

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


def compute_bm25(k1, b, doc, query, idf, avg_doc_len):
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
        score = compute_bm25(1.5, 0.75, sample_tokens, log_tokens, idf, avg_doc_len)
        scores.append(score)

    return np.array(scores)


def find_top_N_similar_logs(current_log, sample_df, N, permutation):
    sample_contents = sample_df['Content'].values
    sample_templates = sample_df['EventTemplate'].values

    # 计算IDF和平均文档长度
    sample_tokens_list = [log.split() for log in sample_contents]
    avg_doc_len = np.mean([len(tokens) for tokens in sample_tokens_list])
    idf = compute_idf(sample_tokens_list)

    list_contents = []
    list_templates = []

    scores = bm25_similarity(current_log, sample_contents, idf, avg_doc_len)

    #ascend
    if permutation == 'ascend':
        top_N_indices = np.argsort(scores)[-N:]
    #descend
    elif permutation == 'descend':
        top_N_indices = np.argsort(scores)[-N:][::-1]

    # print(scores[top_N_indices])

    top_N_logs = sample_contents[top_N_indices]
    top_N_templates = sample_templates[top_N_indices]
    for log, template in zip(top_N_logs, top_N_templates):
        list_contents.append(log)
        list_templates.append(template)

    return list_contents, list_templates

def sampleLogs(df, sample_ratio):

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
        smoothing_factor = (max_complexity - min_complexity) * 0.2

        # weights = cluster_data['complexity_score'] / cluster_data['complexity_score'].sum()
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

class ModelParser():
    def __init__(self,
                 log_path,
                 result_path,
                 dataset,
                 permutation,
                 sample_ratio,
                 model_path,
                 evaluate,  # evaluate or not
                 ):

        self.log_path = log_path + "/{}/{}_2k.log_structured.csv".format(dataset, dataset)
        self.result_path = result_path
        self.dataset = dataset
        self.permutation = permutation
        self.sample_ratio = sample_ratio
        self.model_path = model_path
        self.evaluate = evaluate

    # extract result from model's response
    def extractResultTemplate(self, text):
        # this pattern is for ChatGPT
        # pattern = re.compile('<START> <Event\d> (.+) <END>')

        # output_str = re.sub(r'^log parsed template: ', '', text).rstrip()
        # return output_str
        pattern = re.compile('<START> (.+) <END>')
        result = pattern.findall(text)
        if (len(result)):
            return result[0]
        else:
            return ""

    # def BatchParse(self, model_unuse, model_name, limit, N=5):
    def BatchParse(self, model, limit, N=5):
        # The following 10 lines setting will lead to FP16 based llm
        device = torch.device('cuda:0')

        model_path = self.model_path
        print(model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).half().to(device)

        answer_list = []
        instruction = "For each log after <content> tag, try your best to extract one log template\
        (substitute variable tokens in the log as <*> and remain constant tokens to construct the template)\
        and put the template after <template> tag and between <START> and <END> tags."
        txt_path = self.result_path + f"/{self.dataset}_runtime_error.txt"
        self.result_path = self.result_path + "/{}_{}_result.csv".format(limit, self.dataset)
        # if the result file already exists, load it
        if os.path.exists(self.result_path):
            print("Result file already exists, loading ...")
            answer_list = pd.read_csv(self.result_path)['template'].to_list()
        else:
            # if the result file does not exist, use api to generate result
            print("Result file does not exist, generating result ...")
            df = pd.read_csv(self.log_path)
            contents = df['Content'].tolist()
            sample_df = sampleLogs(df, self.sample_ratio)
            start_time = time.time()

            for log in tqdm(contents):
                gt = df.loc[df['Content'] == log, 'EventTemplate'].values[0]
                similar_contents, similar_templates = find_top_N_similar_logs(log, sample_df, N, self.permutation)
                prompt = "\n"
                for i in range(N):
                    prompt = (prompt + "<content>: " + similar_contents[i] + "\n"
                              + "<template>: <START> " + similar_templates[i] + " <END>\n")
                prompt = prompt + "<content>: " + log + "\n<template>: "
                # print(prompt)
                cnt = 0
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt}
                ]

                while True:
                    inputs = tokenizer.apply_chat_template(messages,
                                                           add_generation_prompt=True,
                                                           tokenize=True,
                                                           return_tensors="pt",
                                                           return_dict=True
                                                           )
                    max_len = sum(len(message["content"]) for message in messages)
                    # print(max_len)
                    inputs = inputs.to(device)

                    gen_kwargs = {"max_length": max_len, "do_sample": True, "top_k": 1}
                    with torch.no_grad():
                        outputs = model.generate(**inputs, **gen_kwargs)
                        outputs = outputs[:, inputs['input_ids'].shape[1]:]
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # print(response)
                    cnt = cnt + 1
                    result = ""
                    if response is not None:
                        result = self.extractResultTemplate(response)
                    # print(result)
                    if result != "":
                        checkResult, errors, replacements = validate_log_transformation(log, result)
                        if checkResult:
                            if len(result.split()) == 1:  # 讨论len==1的result
                                if len(log.split()) <= 4:
                                    answer_list.append(result)
                                    break
                            else:
                                answer_list.append(result)
                                break
                    if cnt >= 5:
                        with open(txt_path, 'a') as file:
                            # 写入数据
                            file.write('-' * 50)
                            file.write("\n")
                            file.write(prompt)
                            file.write("\n")
                            file.write("gt: " + gt)
                            file.write("\n")
                            file.write("parsed result: " + result)
                            file.write("\n")
                        # print('-'*50)
                        # print(prompt)
                        # print("gt: " + templates[i])
                        # print("parsed result: " + result)
                        # 太多次找不到合适 选择相近的template
                        if self.permutation == 'ascend':
                            result = similar_templates[-1]
                        elif self.permutation == 'descend':
                            reverse = similar_templates[0]
                        print("similar temp: " + result)
                        answer_list.append(result)
                        break

            end_time = time.time()
            print(f"{self.dataset} total time: {end_time - start_time}")
            print(f"average time: {(end_time - start_time) / limit}")
            with open(txt_path, 'a') as file:
                file.write(f"{self.dataset} total time: {end_time - start_time}")
                file.write("\n")
                file.write(f"average time: {(end_time - start_time) / limit}")
                file.write("\n")

            print("Writing result into {} ...".format(self.result_path))
            print(len(contents))
            print(len(answer_list))
            if not os.path.exists(self.result_path):
                output = pd.DataFrame(data={"log": contents, "template": answer_list})
                output.to_csv(self.result_path, index=False)
            print("Result file generated.")

        return
