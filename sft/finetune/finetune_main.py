# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
import os
import sys
from pathlib import Path
DIR = Path(os.path.realpath(os.path.dirname(__file__))).parent
sys.path.insert(0, str(DIR))
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

import transformers
from accelerate.utils import DistributedType
from transformers.integrations import deepspeed 
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model
from transformers.trainer_pt_utils import LabelSmoother
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Trainer, set_seed, TrainingArguments
from qwen_vl_utils import process_vision_info
from functools import partial

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
set_seed(42)

class DramaData(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = json.load(open(data_path))
                

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='')

@dataclass
class DataArguments:
    data_path: str = field(
        default='data/mdrama_sft.json', metadata={'help': 'Path to the training data json (produced by data/make_sft_data.py).'})
    batch_size: int = 1

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    max_length: int = field(
        default=8192,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
    use_lora: bool = False
    tune_llm: bool = False
    tune_vision: bool = False
    tune_mlp: bool = False
    tune_llm_head: bool = False
    dataloader_num_workers: int = 2
    remove_unused_columns: bool = False
    attn_implementation: str = 'sdpa'


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj"
    ])
    lora_weight_path: str = ''
    lora_bias: str = 'none'

def maybe_zero_3(param):
    if hasattr(param, 'ds_id'):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == 'none':
        to_return = {k: t for k, t in named_params if 'lora_' in k}
    elif bias == 'all':
        to_return = {
            k: t
            for k, t in named_params if 'lora_' in k or 'bias' in k
        }
    elif bias == 'lora_only':
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if 'lora_' in k:
                to_return[k] = t
                bias_name = k.split('lora_')[0] + 'bias'
                lora_bias_names.add(bias_name)
            elif 'bias' in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str,
                                   bias='none'):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict(
        )
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias)
        else:
            state_dict = trainer.model.state_dict()

    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    collate_fn
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = DramaData(data_args.data_path)
    print(str(len(train_dataset)) + 'samples is loaded')
    eval_dataset = None

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

def find_assistant_content_sublist_indexes(l):
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant\n")
    # [151644, 77091, 198]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>\n")
    # [151645, 198]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 2):
        # Check if the current and next elements form the start sequence
        if l[i] == 151644 and l[i+1] == 77091 and l[i+2] == 198:
            start_indexes.append(i+3)
            # Now look for the first 151645 and 198 after the start
            for j in range(i+3, len(l)-1):
                if l[j] == 151645 and l[j+1] == 198:
                    end_indexes.append(j+2) # **NOTE** the <|im_end|>\n 2 tokens should be included in the label, so that model can predicate end of output.
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments))

    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    if local_rank == 0:
        print(training_args)

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    processor.video_processor.do_sample_frames = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        padding_side='right', 
        model_max_length=training_args.max_length
    )

    def collate_fn(batch, processor):
    
        messages = [m['messages'] for m in batch]

        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
        image_inputs, video_inputs = process_vision_info(messages, return_video_metadata=True)

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs[0][0],
            video_metadata=video_inputs[0][1],
            padding=True,
            return_tensors="pt",
        )

        input_ids_lists = inputs['input_ids'].tolist()
        assert len(messages) == len(input_ids_lists)

        labels_list = []
        for ids_list in input_ids_lists:
            label_ids = [-100] * len(ids_list)
            for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
                label_ids[begin_end_indexs[0]:begin_end_indexs[1]] = ids_list[begin_end_indexs[0]:begin_end_indexs[1]]
            labels_list.append(label_ids)

        labels = torch.tensor(labels_list, dtype=torch.int64)
        inputs['labels'] = labels

        return inputs

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    config.max_length = training_args.max_length

    # Load model and tokenizer
    print(f'Load model from: {model_args.model_name_or_path}')

    if "qwen3" in model_args.model_name_or_path.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
            attn_implementation=training_args.attn_implementation
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=torch.float16 if training_args.fp16 else torch.bfloat16,
            attn_implementation=training_args.attn_implementation
        )

    model.tokenizer = tokenizer

    # freeze all params
    for name, param in model.named_parameters():
        param.requires_grad = False

    def _active_params(module):
        for param in module.parameters():
            param.requires_grad = True

    if training_args.tune_vision:
        _active_params(model.visual)

    if training_args.tune_llm:
        _active_params(model.language_model)

    if training_args.tune_mlp:
        _active_params(model.visual.merger)

    if training_args.tune_llm_head:
        _active_params(model.lm_head)

    if training_args.use_lora:
        for name, param in model.named_parameters():
            param.requires_grad = False
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type='CAUSAL_LM',
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, collate_fn=partial(collate_fn, processor=processor))
    print(transformers.processing_utils.logging.is_progress_bar_enabled())
    transformers.processing_utils.logging.enable_progress_bar()

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        bias=lora_args.lora_bias)


if __name__ == '__main__':
    train()
