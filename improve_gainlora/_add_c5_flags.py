"""Add --run_single False and --n_batches_c5 100 to task 2+ in gen_scripts."""
import re, os

BASE = '/Users/nnminh322/Desktop/personal/Continual/improve_gainlora'

T5_SCRIPTS = [
    os.path.join(BASE, 'gen_script_superni_order1_t5_specroute.sh'),
    os.path.join(BASE, 'gen_script_superni_order2_t5_specroute.sh'),
    os.path.join(BASE, 'gen_script_long_order3_t5_specroute.sh'),
    os.path.join(BASE, 'gen_script_long_order4_t5_specroute.sh'),
]

for name in T5_SCRIPTS:
    if not os.path.exists(name):
        print(f'SKIP (not found): {name}')
        continue
    
    with open(name) as f:
        lines = f.readlines()
    
    # Process line by line to find and modify task 2+ blocks
    # Task 2+ blocks have "--per_device_train_batch_size $BSZ" but NOT "--run_single True"
    # We need to add --run_single False and --n_batches_c5 100 before --data_replay_freq
    
    result = []
    in_task_ge_2 = False
    for i, line in enumerate(lines):
        # Detect task 1: has "--run_single True"
        if '--run_single True' in line:
            in_task_ge_2 = False
            result.append(line)
        # Detect task 2+: previous task block ended, and this block has "--output_dir" with task number > 1
        elif 'outputs/' in line and re.search(r'/(\d+)-', line):
            task_num = int(re.search(r'/(\d+)-', line).group(1))
            in_task_ge_2 = task_num > 1
            result.append(line)
        # Add flags before --data_replay_freq for task 2+ 
        elif in_task_ge_2 and line.strip() == '--data_replay_freq -1 \\':
            # Check if we need to add the flags
            # Look back to see if --run_single or --n_batches_c5 already present in recent lines
            recent = ''.join(lines[max(0, i-20):i])
            if '--run_single' not in recent and '--n_batches_c5' not in recent:
                result.append('   --run_single False \\\n')
                result.append('   --n_batches_c5 100 \\\n')
            result.append(line)
        else:
            result.append(line)
    
    with open(name, 'w') as f:
        f.writelines(result)
    
    print(f'{os.path.basename(name)}: added C5 flags')

print('Done.')
