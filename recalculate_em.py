import json
import os

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_matrix_from_outputs(base_dir, run_name, tasks, metric='exact_match'):
    matrix = []
    for i in range(len(tasks)):
        row = []
        res_file = f"{base_dir}/{run_name}/outputs/{i+1}-{tasks[i]}/all_results.json"
        if not os.path.exists(res_file):
            matrix.append([0.0]*len(tasks))
            continue
        data = load_json(res_file)
        for j in range(i + 1):
            key = f"predict_{metric}_for_{tasks[j]}"
            row.append(data.get(key, 0.0))
        row.extend([0.0]*(len(tasks)-len(row)))
        matrix.append(row)
    return matrix

def calculate_stats(matrix):
    task_num = len(matrix[0])
    final_row = matrix[-1]
    AP = sum(final_row) / task_num
    
    fgt_list = []
    for j in range(task_num - 1):
        history = [row[j] for row in matrix if row[j] > 0]
        if not history:
            continue
        best = max(history)
        final = final_row[j]
        fgt_list.append(best - final)
    
    Fgt = sum(fgt_list) / len(fgt_list) if fgt_list else 0.0
    return AP, Fgt

tasks = ["yelp", "amazon", "mnli", "cb", "copa", "qqp", "rte", "imdb", "sst2", "dbpedia", "agnews", "yahoo", "multirc", "boolq", "wic"]

# V5 (EM)
v5_dir = "/Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve"
v5_run = "gen_script_long_order3_t5_small_specroute_v5"

print("--- V5 (EM) ---")
try:
    v5_matrix = get_matrix_from_outputs(v5_dir, v5_run, tasks, 'exact_match')
    v5_ap_em, v5_fgt_em = calculate_stats(v5_matrix)
    print(f"V5 AP(EM): {v5_ap_em:.4f}")
    print(f"V5 Fgt(EM): {v5_fgt_em:.4f}")
except Exception as e:
    print(f"V5 failed: {e}")

# ROOT (EM) - Based on User's Markdown since I can't find some ROOT JSONs
# Actually, let's try to parse ROOT logs if any, but 59.7 is definitely the target.
print("--- ROOT (EM) Target ---")
print("ROOT AP(EM): 59.70")

# V10a (EM) - From Log
v10_final_em = {
    "agnews": 38.7237,
    "amazon": 29.0263,
    "boolq": 62.4465,
    "cb": 0.0,
    "copa": 55.0,
    "dbpedia": 40.5395,
    "imdb": 90.0789,
    "mnli": 32.1316,
    "multirc": 59.1172,
    "qqp": 64.3158,
    "rte": 52.7076,
    "sst2": 83.945,
    "wic": 56.4263,
    "yahoo": 64.8947,
    "yelp": 21.3289
}
# Order: yelp, amazon, mnli, cb, copa, qqp, rte, imdb, sst2, dbpedia, agnews, yahoo, multirc, boolq, wic
ordered_v10_em = [21.3289, 29.0263, 32.1316, 0.0, 55.0, 64.3158, 52.7076, 90.0789, 83.9450, 40.5395, 38.7237, 64.8947, 59.1172, 62.4465, 56.4263]
v10_ap_em = sum(ordered_v10_em) / 15
print(f"V10a AP(EM): {v10_ap_em:.4f}")
