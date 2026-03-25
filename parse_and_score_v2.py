import re
import sys

def parse_log(log_path):
    with open(log_path, 'r') as f:
        content = f.read()

    # Task order as defined in script
    tasks = ["yelp", "amazon", "mnli", "cb", "copa", "qqp", "rte", "imdb", "sst2", "dbpedia", "agnews", "yahoo", "multirc", "boolq", "wic"]
    
    # Split content into segments, each ending with a "predict metrics" block
    # We look for "predict_exact_match_for_CL" as an anchor for each step evaluation
    segments = re.split(r'predict_exact_match_for_CL\s+=\s+\d+\.\d+', content)
    
    # The last segment might be empty if there's nothing after the final metrics
    if not segments[-1].strip():
        segments = segments[:-1]
    
    # We expect 15 evaluations
    print(f"Found {len(segments)} evaluation segments in {log_path}")
    
    matrix = []
    for seg in segments:
        scores = []
        for task in tasks:
            match = re.search(fr'predict_exact_match_for_{task}\s+=\s+(\d+\.\d+|\d+)', seg)
            if match:
                scores.append(float(match.group(1)))
            else:
                scores.append(0.0)
        if any(s > 0 for s in scores): # Only add if we found at least one score
            matrix.append(scores)

    # If it's the final evaluation only (like in some logs), we might have only 1 segment
    return matrix, tasks

def calculate_metrics(matrix):
    if not matrix:
        return None
    
    task_num = len(matrix[0])
    # final_scores is the last row provided (if the run ended at 15, it's matrix[14])
    final_scores = matrix[-1]
    AP = sum(final_scores) / task_num
    
    # Forgetting: max(history) - final
    fgt_list = []
    for t_idx in range(task_num - 1):
        history = [row[t_idx] for row in matrix]
        best = max(history)
        final = final_scores[t_idx]
        fgt_list.append(best - final)
    
    Fgt = sum(fgt_list) / len(fgt_list) if fgt_list else 0.0
    
    # User's definition of forgetting in markdown: Final - Initial?
    # Let's calculate that too just in case
    fgt_user_list = []
    for t_idx in range(task_num - 1):
        initial = matrix[t_idx][t_idx] if t_idx < len(matrix) else 0.0
        final = final_scores[t_idx]
        fgt_user_list.append(final - initial)
    Fgt_user = sum(fgt_user_list) / len(fgt_user_list) if fgt_user_list else 0.0

    return {
        "AP": AP,
        "Fgt (Best-Final)": Fgt,
        "Fgt_user (Final-Initial)": Fgt_user,
        "Final Scores": final_scores
    }

log_v10 = "/Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/improve_gainlora_v10.log"
matrix, tasks = parse_log(log_v10)
metrics = calculate_metrics(matrix)
print("--- V10 Metrics ---")
if metrics:
    print(f"AP (EM): {metrics['AP']:.4f}")
    print(f"Fgt (Best-Final): {metrics['Fgt (Best-Final)']:.4f}")
    print(f"Fgt (Final-Initial): {metrics['Fgt_user (Final-Initial)']:.4f}")
    print("Final Scores:", metrics['Final Scores'])
else:
    print("Failed to parse matrix for V10")

# Also do V5 for comparison
log_v5_dir = "/Users/nnminh322/Desktop/personal/Continual/improve_gainlora/logs/t5_small_improve/gen_script_long_order3_t5_small_specroute_v5/"
# We need to find the log file inside v5 dir.
# It's likely in outputs/ or similar.
