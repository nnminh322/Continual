#!/bin/bash
# Check training status for all tasks in a run
# Usage: ./check_task_status.sh <run_name> [output_dir]
# Example: ./check_task_status.sh long_order3_t5_srt logs_and_outputs/long_order3_t5_srt/outputs

RUN_NAME=${1:-long_order3_t5_srt}
OUTPUT_BASE=${2:-logs_and_outputs/$RUN_NAME/outputs}
LOG_FILE=${3:-$OUTPUT_BASE.log}

TASKS=("yelp" "amazon" "mnli" "cb" "copa" "qqp" "rte" "imdb" "sst2" "dbpedia" "agnews" "yahoo" "multirc" "boolq" "wic")

echo "=========================================="
echo "Checking: $RUN_NAME"
echo "Output dir: $OUTPUT_BASE"
echo "=========================================="
echo ""

for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    task_id=$((i + 1))
    task_dir="$OUTPUT_BASE/${task_id}-${task}"
    saved_weights="$task_dir/saved_weights"
    
    # Status indicators
    status="❓ UNKNOWN"
    train_done="❌"
    eval_done="❌"
    predict_done="❌"
    
    # Check if task dir exists
    if [ -d "$task_dir" ]; then
        # Check train checkpoint
        if [ -f "$saved_weights/lora_q.weight" ] || [ -f "$saved_weights/pytorch_model.bin" ]; then
            train_done="✅"
        fi
        
        # Check eval results
        if [ -f "$task_dir/all_results.json" ]; then
            eval_done="✅"
            # Check if eval scores exist for this task
            if grep -q "eval_exact_match_for_$task" "$task_dir/all_results.json" 2>/dev/null; then
                eval_score=$(grep "eval_exact_match_for_$task" "$task_dir/all_results.json" | head -1 | sed 's/.*: *//' | tr -d ' ,')
            else
                eval_score="N/A"
            fi
        else
            eval_score="❌ no file"
        fi
        
        # Check predict results
        if [ -f "$task_dir/predictions.json" ] || grep -q "predict_exact_match_for_$task" "$task_dir/all_results.json" 2>/dev/null; then
            predict_done="✅"
            predict_score=$(grep "predict_exact_match_for_$task" "$task_dir/all_results.json" 2>/dev/null | head -1 | sed 's/.*: *//' | tr -d ' ,')
        else
            predict_score="❌"
        fi
        
        # Determine status
        if [ "$train_done" = "✅" ] && [ "$eval_done" = "✅" ] && [ "$predict_done" = "✅" ]; then
            status="✅ COMPLETE"
        elif [ "$train_done" = "✅" ]; then
            status="⚠️ TRAINED, no eval"
        else
            status="🔄 IN PROGRESS / FAILED"
        fi
        
        # Get latest checkpoint step
        if [ -f "$task_dir/checkpoint-"*/trainer_state.json" ]; then
            latest_step=$(cat "$task_dir"/checkpoint-*/trainer_state.json 2>/dev/null | grep -o '"step":[0-9]*' | tail -1 | cut -d: -f2)
        fi
        
        echo "Task $task_id: $task"
        echo "  Status: $status"
        echo "  Train:  $train_done | Eval: $eval_done | Predict: $predict_done"
        echo "  Eval:   $eval_score | Predict: $predict_score"
        if [ -n "$latest_step" ]; then
            echo "  Step:   $latest_step"
        fi
        echo ""
    else
        echo "Task $task_id: $task"
        echo "  Status: ⏳ NOT STARTED"
        echo ""
    fi
done

echo "=========================================="
echo "Summary"
echo "=========================================="

# Count completed
total=${#TASKS[@]}
completed=$(find "$OUTPUT_BASE" -name "all_results.json" -exec grep -l "eval_exact_match" {} \; 2>/dev/null | wc -l)
with_predict=$(grep -l "predict_exact_match" "$OUTPUT_BASE"/*/all_results.json 2>/dev/null | wc -l)

echo "Total tasks: $total"
echo "With eval results: $completed"
echo "With predict results: $with_predict"
