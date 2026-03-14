from torch import nn
import torch
from typing import Dict

task_config = {
    "task1590_diplomacy_text_generation": "configs/SuperNI/task1590_diplomacy_text_generation",
    "task181_outcome_extraction": "configs/SuperNI/task181_outcome_extraction",
    "task591_sciq_answer_generation": "configs/SuperNI/task591_sciq_answer_generation",
    "task1729_personachat_generate_next": "configs/SuperNI/task1729_personachat_generate_next",
    "task1572_samsum_summary": "configs/SuperNI/task1572_samsum_summary",
    "task1510_evalution_relation_extraction": "configs/SuperNI/task1510_evalution_relation_extraction",
    "task748_glucose_reverse_cause_event_detection": "configs/SuperNI/task748_glucose_reverse_cause_event_detection",
    "task002_quoref_answer_generation": "configs/SuperNI/task002_quoref_answer_generation",
    "task1687_sentiment140_classification": "configs/SuperNI/task1687_sentiment140_classification",
    "task511_reddit_tifu_long_text_summarization": "configs/SuperNI/task511_reddit_tifu_long_text_summarization",
    "task875_emotion_classification": "configs/SuperNI/task511_reddit_tifu_long_text_summarization",
    "task639_multi_woz_user_utterance_generation": "configs/SuperNI/task639_multi_woz_user_utterance_generation",
    "task1290_xsum_summarization": "configs/SuperNI/task1290_xsum_summarization",
    "task073_commonsenseqa_answer_generation": "configs/SuperNI/task073_commonsenseqa_answer_generation",
    "task363_sst2_polarity_classification": "configs/SuperNI/task363_sst2_polarity_classification",
    "dbpedia": "configs/gen_script_long_order3_t5_configs/dbpedia",
    "amazon": "configs/gen_script_long_order3_t5_configs/amazon",
    "agnews": "configs/gen_script_long_order3_t5_configs/agnews",
    "yahoo": "configs/gen_script_long_order3_t5_configs/yahoo",
    "yelp": "configs/gen_script_long_order3_t5_configs/yelp",
    "copa": "configs/gen_script_long_order3_t5_configs/copa",
    "mnli": "configs/gen_script_long_order3_t5_configs/mnli",
    "cb": "configs/gen_script_long_order3_t5_configs/cb",
    "imdb": "configs/gen_script_long_order3_t5_configs/imdb",
    "multirc": "configs/gen_script_long_order3_t5_configs/multirc",
    "sst2": "configs/gen_script_long_order3_t5_configs/sst2",
    "boolq": "configs/gen_script_long_order3_t5_configs/boolq",
    "rte": "configs/gen_script_long_order3_t5_configs/rte",
    "wic": "configs/gen_script_long_order3_t5_configs/wic",
    "qqp": "configs/gen_script_long_order3_t5_configs/qqp",
}

def lora_state_dict_A(model: nn.Module, bias: str = 'none', task_name=None) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_A' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_A' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_A')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError

def lora_state_dict_B(model: nn.Module, bias: str = 'none', task_name=None) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_B' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_B' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_B')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
    
def lora_state_dict_s(model: nn.Module, bias: str = 'none', task_name=None) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_s' in k}
    else:
        raise NotImplementedError