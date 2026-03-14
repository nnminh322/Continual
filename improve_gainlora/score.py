import json
import os
import sys

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def cal_continue_learning_metrics(scores_array, individual_scores):
    task_num = len(scores_array)
    Cl = sum(scores_array[-1]) / task_num

    fgt_list = []
    for t_idx in range(task_num - 1):
        history = [line[t_idx] for line in scores_array[:-1]]
        history_best = max(history)
        fgt_list.append(history_best - scores_array[-1][t_idx])
    Fgt = sum(fgt_list) / len(fgt_list)

    Fwt = sum([scores_array[i][i] for i in range(task_num)]) / task_num - 0
    Bwt = sum([scores_array[-1][i] - scores_array[i][i] for i in range(task_num)]) / task_num

    return {'Cl': Cl, 'Fgt': Fgt, 'Fwt': Fwt, 'Bwt': Bwt}


def find_base_dir(run_name):
    """Auto-detect the logs_and_outputs base directory."""
    if len(sys.argv) >= 4:
        return sys.argv[3]

    candidates = [
        "logs_and_outputs",
        "../logs_and_outputs",
        "/kaggle/working/Continual/root_gainlora/logs_and_outputs",
        "/kaggle/working/logs_and_outputs",
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, run_name)):
            return c

    # Search relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fallback = os.path.join(script_dir, "logs_and_outputs")
    if os.path.isdir(os.path.join(fallback, run_name)):
        return fallback

    print(f"[ERROR] Cannot find logs_and_outputs/{run_name} from any known path.")
    print(f"  Tried: {candidates}")
    print(f"  Usage: python score.py <run_name> <single_path> [base_dir]")
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print("Usage: python score.py <run_name> <single_path> [base_dir]")
        sys.exit(1)

    run_name = sys.argv[1]
    single_path = sys.argv[2]
    base_dir = find_base_dir(run_name)

    print(f"[INFO] base_dir: {base_dir}")
    print(f"[INFO] run_name: {run_name}")

    task_order_file = os.path.join(base_dir, run_name, "outputs", "task_order.txt")
    if not os.path.exists(task_order_file):
        print(f"[ERROR] task_order.txt not found: {task_order_file}")
        print("  Make sure bash script includes --do_predict flag for each task.")
        sys.exit(1)

    with open(task_order_file, 'r') as f:
        data_list = f.read().strip().split(',')

    task_num = len(data_list)
    result_root_path = os.path.join(base_dir, run_name, "outputs")
    single_root_path = os.path.join(base_dir, single_path, "outputs")

    # Build Cross-Task Score Matrix
    scores = []
    missing_predict = []
    for i in range(task_num):
        score_line = []
        res_file = os.path.join(result_root_path, f'{i+1}-{data_list[i]}', 'all_results.json')
        if not os.path.exists(res_file):
            print(f"[WARN] Missing result file: {res_file}")
            missing_predict.append(i)
            scores.append([0.0] * task_num)
            continue
        inference_result = load_json(res_file)
        for j in range(i + 1):
            if 'superni' in run_name:
                key = f'predict_eval_rougeL_for_{data_list[j]}'
            else:
                key = f'predict_exact_match_for_{data_list[j]}'
            score = inference_result.get(key, None)
            if score is None:
                print(f"[WARN] Key '{key}' not in {res_file}. Was --do_predict missing?")
                score = 0.0
            score_line.append(score)
        score_line.extend([0.0] * (task_num - i - 1))
        scores.append(score_line)

    if missing_predict:
        print(f"[WARN] {len(missing_predict)} tasks missing predict results. FT/Fgt may be inaccurate.")

    # Single-task baseline scores
    single_order_file = os.path.join(single_root_path, "task_order.txt")
    if not os.path.exists(single_order_file):
        print(f"[WARN] single task_order.txt not found. Using same task order.")
        single_task_list = data_list
    else:
        with open(single_order_file, 'r') as f:
            single_task_list = f.read().strip().split(',')

    individual_scores = []
    for i in range(task_num):
        res_file = os.path.join(single_root_path, f'{i+1}-{single_task_list[i]}', 'all_results.json')
        if not os.path.exists(res_file):
            print(f"[WARN] Missing single result: {res_file}. Using 0.0.")
            individual_scores.append(0.0)
            continue
        inference_result = load_json(res_file)
        if 'superni' in run_name:
            key = f'predict_eval_rougeL_for_{single_task_list[i]}'
        else:
            key = f'predict_exact_match_for_{single_task_list[i]}'
        individual_scores.append(inference_result.get(key, 0.0))

    # Compute metrics
    cl_scores = cal_continue_learning_metrics(scores, individual_scores)
    print(json.dumps(cl_scores, indent=2))

    avg_scores = [sum(score[:i+1])/(i+1) for i, score in enumerate(scores)]

    try:
        from tabulate import tabulate
        title = list(range(task_num))
        print(tabulate([individual_scores], headers=title, tablefmt='fancy_grid'))

        # Ensure results directory exists
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        with open(os.path.join(results_dir, run_name + '.txt'), 'w') as f:
            f.write(str(cl_scores) + '\n')
            f.write(tabulate([individual_scores], headers=title, tablefmt='fancy_grid') + '\n')
            title2 = [''] + list(range(task_num))
            scores_line = [[i] + line for i, line in enumerate(scores)]
            print(tabulate(scores_line, headers=title2, tablefmt='fancy_grid'))
            f.write(tabulate(scores_line, headers=title2, tablefmt='fancy_grid'))
        print(f"[INFO] Results saved to results/{run_name}.txt")
    except ImportError:
        print("[WARN] tabulate not installed, skipping table formatting.")
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, run_name + '.txt'), 'w') as f:
            f.write(str(cl_scores) + '\n')
            f.write(str(individual_scores) + '\n')

    print("avg_scores:", avg_scores)


if __name__ == "__main__":
    main()