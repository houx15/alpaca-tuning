import os
import sys

import argparse

import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import pyarrow as pa
from tqdm import tqdm

# import peft
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import (
    prepare_model_for_int8_training,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error

import json

# NOTE: deprecated in python 3.10
from collections import Iterable

from utils import *


class MyDataset(Dataset):
    r'''
    # TODO: This class was created due to only the restriction of some packages in colab. Maybe we need to make it more concise.
    '''
    def __init__(self, arrow_table: pa.Table):
        self._data: pa.Table = arrow_table
        pass

    @staticmethod
    def _unnest(py_dict):
        return dict((key, array[0]) for key, array in py_dict.items())
    
    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key = self._data.num_rows + key
            if key >= self._data.num_rows:
                raise IndexError(f"Index ({key}) outside of table length ({self._data.num_rows}).")
            outputs = self._unnest(self._data.slice(key, 1).to_pydict())
        elif isinstance(key, slice):
            key_indices = key.indices(self._data.num_rows)
            if key_indices[2] != 1 or key_indices[1] < key_indices[0]:
                raise ValueError("Slicing can only take contiguous and ordered slices.")
            outputs = self._data.slice(key_indices[0], key_indices[1] - key_indices[0]).to_pydict()
        elif isinstance(key, str):
            if key not in self._data.column_names:
                raise ValueError(f"Column ({key}) not in table columns ({self._data.column_names}).")
            outputs = self._data[key].to_pylist()
        elif isinstance(key, Iterable):
            data_subset = pa.concat_tables(self._data.slice(int(i), 1) for i in key)
            outputs = data_subset.to_pydict()

        else:
            raise ValueError("Can only get row(s) (int or slice or list[int]) or columns (string).")
        return outputs
    
    def __len__(self):
        return self._data.num_rows

    @classmethod
    def from_dict(cls, data: dict) -> "MyDataset":
        pa_table: pa.Table = pa.Table.from_pydict(
            mapping=data, schema=None
        )
        return cls(pa_table)


class PromptTune(object):
    r'''
    Evaluate models, including raw models
    TODO Document
    TODO This class can absolutely be aggregated with ModelEval
    '''

    def __init__(self, topic, model_abbr, eight_bit) -> None:
        self.topic = topic

        self.translator = None

        self.load_model_and_tokenizer(model_abbr, eight_bit)
        self.load_result_translator()
        self.load_dataset()
        self.load_prompt_designer()
        
        self.prepare_model_for_train()

        pass

    def load_model_and_tokenizer(self, model_abbr, eight_bit) -> None:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/model_map.json'), 'r', encoding='utf8') as rfile:
            model_dict = json.loads(rfile.read())
            self.model_name = model_dict[model_abbr]
        
        '''
        #TODO 
        - Support more LLMs
        - different scheme to load tokenizer and model
        '''
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name,
            load_in_8bit=eight_bit,
            # NOTE: device map is disabled due to a bug in huggingface.
            # device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        # NOTE: unnecessary when device map is enabled
        self.model.cuda()
        return
    
    def load_dataset(self) -> None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'dataset/{self.topic}')
        self.train_df = pd.read_csv(os.path.join(data_dir, 'train-regression.csv'), header=0, usecols=['text', 'label'])
        self.test_df = pd.read_csv(os.path.join(data_dir, 'test-regression.csv'), header=0, usecols=['text', 'label'])
        self.val_df = pd.read_csv(os.path.join(data_dir, 'validate-regression.csv'), header=0, usecols=['text', 'label'])

        # self.df = pd.read_csv(os.path.join(data_dir, 'total-regression.csv'), header=0, usecols=['text', 'label'])

        # prompt based
        convert_label_dict = {}
        for k, v in self.translator['label'].items():
            convert_label_dict[v] = k
        
        self.train_df['label'] = self.train_df['label'].map(convert_label_dict)
        self.val_df['label'] = self.val_df['label'].map(convert_label_dict)
        self.test_df['label'] = self.test_df['label'].map(convert_label_dict)
        # self.df['label'] = self.df['label'].map(convert_label_dict)


    def load_result_translator(self) -> None:
        if not self.translator:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/label_map.json'), 'r', encoding='utf8') as rfile:
                translator_dict = json.loads(rfile.read())
                self.translator = translator_dict[self.topic]
        return
    

    def load_prompt_designer(self) -> None:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/prompt_design.json'), 'r', encoding='utf8') as rfile:
            prompt_dict = json.loads(rfile.read())
            self.prompt_designer = prompt_dict[self.topic]
        return
    
    
    def prepare_model_for_train(self) -> None:
        NOTFIRSTRUN=False
        if not NOTFIRSTRUN:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2

            self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
        NOTFIRSTRUN=True


    def generate_prompt(self, single_text):
        single_prompt = f'### Instruction: \n{self.prompt_designer}\n### Input:\n{single_text}\n### Response:'
        return single_prompt
    

    def tokenize_and_gen_dataset(self, texts, labels, cutoff_len, add_eos_token=True):
        input_ids = []
        attention_mask = []
        tokenized_labels = []

        for idx, single_text in enumerate(texts):
            single_prompt = self.generate_prompt(single_text)
            result = self.tokenizer(
                single_text,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
            ):
                result["input_ids"].append(self.tokenizer.eos_token_id)
                result["attention_mask"].append(1)

            user_prompt_len = len(result["input_ids"])

            label_result = self.tokenizer(labels[idx], padding=False, return_tensors=None)
            cur_label_ids = label_result["input_ids"]
            cur_label_ids = cur_label_ids + [-100] * (user_prompt_len - len(cur_label_ids))

            input_ids.append(result["input_ids"])
            attention_mask.append(result["attention_mask"])
            tokenized_labels.append(cur_label_ids)

        all_res = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": tokenized_labels
        }
        print(len(all_res["input_ids"][0]), len(all_res["labels"][0]))

        dataset = MyDataset.from_dict(all_res)

        return dataset


    def train(
        self,
        train_df,
        val_df,
        output_dir: str,
        # training hyperparams
        batch_size: int = 32,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    ):
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        )
        gradient_accumulation_steps = batch_size // micro_batch_size

        self.tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        self.tokenizer.padding_side = "left"  # Allow batched inference

        # def generate_and_tokenize_prompt(data_point):
        #     full_prompt = generate_prompt(data_point)
        #     tokenized_full_prompt = tokenize(full_prompt)
        #     if not train_on_inputs:
        #         user_prompt = generate_prompt({**data_point, "output": ""})
        #         tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        #         user_prompt_len = len(tokenized_user_prompt["input_ids"])

        #         tokenized_full_prompt["labels"] = [
        #             -100
        #         ] * user_prompt_len + tokenized_full_prompt["labels"][
        #             user_prompt_len:
        #         ]  # could be sped up, probably
        #     return tokenized_full_prompt

        model = prepare_model_for_int8_training(self.model)

        model.enable_input_require_grads()

        '''
        TODO: configurized the LORA WEIGHTS
        '''
        LORA_WEIGHTS = "tloen/alpaca-lora-7b"
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            torch_dtype=torch.float16,
            is_trainable=True
        )

        # if resume_from_checkpoint:
        #     # Check the available weights and load them
        #     checkpoint_name = os.path.join(
        #         resume_from_checkpoint, "pytorch_model.bin"
        #     )  # Full checkpoint
        #     if not os.path.exists(checkpoint_name):
        #         checkpoint_name = os.path.join(
        #             resume_from_checkpoint, "adapter_model.bin"
        #         )  # only LoRA model - LoRA config above has to fit
        #         resume_from_checkpoint = False  # So the trainer won't try loading its state
        #     # The two files above have a different name depending on how they were saved, but are actually the same.
        #     if os.path.exists(checkpoint_name):
        #         print(f"Restarting from {checkpoint_name}")
        #         adapters_weights = torch.load(checkpoint_name)
        #         model = set_peft_model_state_dict(model, adapters_weights)
        #     else:
        #         print(f"Checkpoint {checkpoint_name} not found")

        model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

        # if val_set_size > 0:
        #     train_val = data["train"].train_test_split(
        #         test_size=val_set_size, shuffle=True, seed=42
        #     )
        #     train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        #     val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        # else:
        #     train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        #     val_data = None
        train_df = train_df.sample(frac=1)
        val_df = val_df.sample(frac=1)

        train_data = self.tokenize_and_gen_dataset(train_df.text.values, train_df.label.values, cutoff_len)
        val_data = self.tokenize_and_gen_dataset(val_df.text.values, val_df.label.values, cutoff_len)

        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                fp16=False,
                logging_steps=10,
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=200 if val_set_size > 0 else None,
                save_steps=200,
                output_dir=output_dir,
                save_total_limit=3,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=None,
                group_by_length=group_by_length,
            ),
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        model.config.use_cache = False

        # old_state_dict = model.state_dict
        # model.state_dict = (
        #     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        # ).__get__(model, type(model))

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        model.save_pretrained(output_dir)

        print("\n If there's a warning about missing keys above, please disregard :)")
    

    def run_train(self, output_dir) -> None:
        self.train(self.train_df, self.val_df, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction/Evaluation.')
    
    parser.add_argument('topic', type=str, choices=['gun', 'china', 'abortion', 'drug', 'climate', 'sexual'])
    parser.add_argument('--model', type=str, choices=['guanaco', 'alpaca', 'alpaca-13b', 'falcon'])
    parser.add_argument('--eight_bit', type=bool, default=True)
    parser.add_argument('--output_dir', type=str, default='./')

    config = parser.parse_args()

    evaluator = PromptTune(topic=config.topic, model_abbr=config.model, eight_bit=config.eight_bit)
    evaluator.run_train(config.output_dir)


