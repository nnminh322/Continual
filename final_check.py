import json
import os

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_matrix_from_outputs(base_dir, run_name, tasks):
    matrix = []
    for i in range(len(tasks)):
        row = []
        res_file = f"{base_dir}/{run_name}/outputs/{i+1}-{tasks[i]}/all_results.json"
        if not os.path.exists(res_file):
            matrix.append([0.0]*len(tasks))
            continue
        data = load_json(res_file)
        for j in range(i + 1):
            key = f"predict_eval_rougeL_for_{tasks[j]}"
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

# ROOT
root_dir = "/Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/root_t5_small"
root_run = "gen_script_long_order3_t5_small_gainlora_inflora"
# ROOT might not have all_results.json with predict metrics as seen earlier. 
# So I'll use the user's documented values for ROOT if needed.
# But let's try reading V5 which definitely has them.
v5_dir = "/Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve"
v5_run = "gen_script_long_order3_t5_small_specroute_v5"

print("--- V5 Matrix ---")
try:
    v5_matrix = get_matrix_from_outputs(v5_dir, v5_run, tasks)
    v5_ap, v5_fgt = calculate_stats(v5_matrix)
    print(f"V5 AP(rougeL): {v5_ap:.4f}")
    print(f"V5 Fgt: {v5_fgt:.4f}")
except Exception as e:
    print(f"V5 failed: {e}")

# For V10, we have the final vector from log:
v10_final = [59.9013, 59.7018, 30.5395, 0.0, 55.0, 11.9474, 10.1083, 89.8947, 65.2523, 53.1737, 65.0342, 62.0329, 43.1312, 62.4465, 56.4263]
v10_ap = sum(v10_final) / 15
print(f"V10 AP(rougeL): {v10_ap:.4f}")
