import os
import argparse
from MicLog_DBSCAN_glm import ModelParser

def main(args):
    # get a tester object with data
    print("Parsing " + args.dataset + " ...")

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    if not os.path.exists(args.log_path):
        print("Log path does not exist. Please check the path.")
        exit()

    parser = ModelParser(
                log_path = args.log_path,        # .log_structured_csv
                result_path=args.result_path,    # .result_csv
                dataset = args.dataset,             # 16 datasets
                permutation=args.permutation,
                sample_ratio=args.sample_ratio,
                model_path=args.model_path,
                evaluate=args.evaluate,  # evaluate or not
                )

    parser.BatchParse(model = args.model,
                      limit = args.limit,         # number of logs for testing
                      N = args.N,                  # number of examples in the prompt
                      )

if __name__ == '__main__':
    # log_list = ['HDFS']
    # log_list = ['Proxifier']
    log_list = ['Android', 'Apache', 'BGL', 'Hadoop', 'HDFS', 'HealthApp', 'HPC', 'Linux',
                'Mac', 'OpenSSH', 'OpenStack', 'Proxifier', 'Spark', 'Thunderbird', 'Windows', 'Zookeeper',]


    sample_ratios = [0.15]
    test_cnts = [1]
    model = '' # meta-trained model name
    model_path = f'/root/autodl-tmp/' + model

    for test_cnt, sample_ratio in zip(test_cnts, sample_ratios):
        result_path = f'MicLog_DBSCAN_{model}_FP16_results_test{test_cnt}_sample_ratio{sample_ratio}'
        for dataset in log_list:
            print(result_path)
            parser = argparse.ArgumentParser()
            parser.add_argument('--log_path', type=str, default='data/loghub_2k', help='log path')
            # parser.add_argument('--result_path', type=str, default='MicLog_final_Qwen2-7B-Instruct_FP16_results_test4', help='result path')
            parser.add_argument('--result_path', type=str, default=result_path, help='result path')
            parser.add_argument('--dataset', type=str, default=dataset, help='dataset name')
            parser.add_argument('--permutation', type=str, default='ascend', help='ascend, descend')
            parser.add_argument('--model', type=str, default=model, help='model name')
            parser.add_argument('--model_path', type=str, default=model_path, help='model path')
            parser.add_argument('--limit', type=int, default=2000, help='number of logs for testing')
            parser.add_argument('--N', type=int, default=5, help='number of examples in the prompt')
            parser.add_argument('--sample_ratio', type=float, default=sample_ratio, help='0.1, 0.15, 0.2,...')
            parser.add_argument('--evaluate', type=bool, default=False, help='evaluate or not')
            args = parser.parse_args()
            main(args)

