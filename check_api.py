import transformers

# Check Trainer methods
t = transformers.Trainer
for item in ['_wrap_model', 'floating_point_ops', 'get_model_param_count',
             'nested_truncate', 'denumpify_detensorize', '_pad_across_processes',
             'ShardedDDPOption', 'is_torch_tpu_available']:
    has = hasattr(t, item)
    print(f"  Trainer.{item}: {'OK' if has else 'MISSING'}")

print()
import transformers.trainer_pt_utils as tpu
for fn in ['nested_truncate', 'denumpify_detensorize', 'IterableDatasetShard',
           'get_model_param_count']:
    has = hasattr(tpu, fn)
    print(f"  trainer_pt_utils.{fn}: {'OK' if has else 'MISSING'}")

print()
ta = transformers.TrainingArguments
print(f"  TrainingArguments.past_index: {'OK' if hasattr(ta, 'past_index') else 'MISSING'}")

print()
import transformers.trainer_utils as tu
for fn in ['denumpify_detensorize', 'get_model_param_count', 'nested_truncate']:
    has = hasattr(tu, fn)
    print(f"  trainer_utils.{fn}: {'OK' if has else 'MISSING'}")
