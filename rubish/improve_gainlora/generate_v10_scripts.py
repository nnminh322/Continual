import re
import os

with open("T5_small/gen_script_long_order3_t5_small_gainlora_inflora.sh", "r") as f:
    gainlora_content = f.read()

with open("T5_small/gen_script_long_order3_t5_small_specroute.sh", "r") as f:
    specroute_content = f.read()

def create_script(mode, suffix):
    new_content = specroute_content.replace("gen_script_long_order3_t5_small_specroute", f"gen_script_long_order3_t5_small_specroute_{suffix}")
    new_content = new_content.replace("--model_name specroute \\", f"--model_name specroute \\\n   --routing_mode {mode} \\")
    
    if mode == "learned":
        # Extract previous_prompt_key_path and load_checkpoint_from from gainlora
        blocks = new_content.split("python src/run_t5.py")
        final_content = blocks[0]
        
        gainlora_blocks = gainlora_content.split("python src/run_t5.py")
        
        for i in range(1, len(blocks)):
            block = blocks[i]
            gainlora_block = gainlora_blocks[i]
            
            m1 = re.search(r'--load_checkpoint_from\s+([^\s\\]+)', gainlora_block)
            m2 = re.search(r'--previous_prompt_key_path\s+([^\s\\]+)', gainlora_block)
            
            args_to_add = ""
            if m1:
                path1 = m1.group(1).replace("gen_script_long_order3_t5_small_gainlora_inflora", "gen_script_long_order3_t5_small_specroute_v10a")
                args_to_add += f"   --load_checkpoint_from {path1} \\\n"
            if m2:
                path2 = m2.group(1).replace("gen_script_long_order3_t5_small_gainlora_inflora", "gen_script_long_order3_t5_small_specroute_v10a")
                args_to_add += f"   --previous_prompt_key_path {path2} \\\n"
                
            final_content += "python src/run_t5.py" + block.replace("   --do_train \\\n", f"   --do_train \\\n{args_to_add}")
            
        new_content = final_content

    with open(f"T5_small/gen_script_long_order3_t5_small_specroute_{suffix}.sh", "w") as f:
        f.write(new_content)
    print(f"Created T5_small/gen_script_long_order3_t5_small_specroute_{suffix}.sh")

create_script("learned", "v10a")
create_script("grassmann", "v10b")
