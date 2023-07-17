import os
import sys

import argparse

import torch

import pandas as pd
import numpy as np
from tqdm import tqdm

# import peft
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error

import json

from utils import *


class ModelEval(object):
    r'''
    Evaluate models, including raw models
    #TODO Document
    '''

    def __init__(self, topic, model_abbr, eight_bit) -> None:
        self.topic = topic

        self.load_model_and_tokenizer(model_abbr, eight_bit)
        self.load_result_translator()
        self.load_dataset()
        self.load_prompt_designer()
        
        self.prepare_model_for_eval()

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
            device_map='auto',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        return
    
    def load_dataset(self) -> None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'dataset/{self.topic}')
        # self.train_df = pd.read_csv(os.path.join(data_dir, 'train-regression.csv'), header=0, usecols=['text', 'label'])
        # self.test_df = pd.read_csv(os.path.join(data_dir, 'test-regression.csv'), header=0, usecols=['text', 'label'])
        # self.val_df = pd.read_csv(os.path.join(data_dir, 'validate-regression.csv'), header=0, usecols=['text', 'label'])

        self.df = pd.read_csv(os.path.join(data_dir, 'total-regression.csv'), header=0, usecols=['text', 'label'])

        # prompt based
        convert_label_dict = {}
        for k, v in self.translator['label'].items():
            convert_label_dict[v] = k
        
        # self.train_df['label'] = self.train_df['label'].map(convert_label_dict)
        # self.val_df['label'] = self.val_df['label'].map(convert_label_dict)
        # self.test_df['label'] = self.test_df['label'].map(convert_label_dict)
        self.df['label'] = self.df['label'].map(convert_label_dict)


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
    
    
    def prepare_model_for_eval(self) -> None:
        NOTFIRSTRUN=False
        if not NOTFIRSTRUN:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2

            self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
        NOTFIRSTRUN=True

    
    def evaluate(
            self,
            prompt='',
            temperature=0.4,
            top_p=0.65,
            top_k=35,
            repetition_penalty=1.1,
            max_new_tokens=512,
            **kwargs,
        ):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda:0")
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )
        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        return output.split("### Response:")[-1].strip()

    def eval(self, texts, labels):
        prediction = np.array([])
        true = np.array([])
        print('evaluate begin...')
        for idx, single_text in tqdm(np.ndenumerate(texts)):
            single_prompt = f'### Instruction: \n{self.prompt_designer}\n### Input:\n{single_text}\n### Response:'
            result = self.evaluate(single_prompt)
            output = result_translator(self.topic, result, self.translator)
            prediction = np.append(prediction, output)
            label = result_translator(self.topic, labels[idx], self.translator)
            true = np.append(true, label)

        acc = accuracy_score(true, prediction)
        data = pd.DataFrame(data={'predict': prediction, 'true': true})
        irrelevant_eval = data.replace({2: 1, 1: 1, 0: 1, -1: 1, -2: 1, -9: 0})
        relevant_data = data.drop(data[(data['true'] == -9) | (data['predict'] == -9)].index)

        relevant_acc = accuracy_score(irrelevant_eval.true.values, irrelevant_eval.predict.values)
        precision, recall, f1, _ = precision_recall_fscore_support(irrelevant_eval.true.values, irrelevant_eval.predict.values, average='binary')

        rmse = mean_squared_error(relevant_data.true.values, relevant_data.predict.values, squared=False)
        error_analysis = {}
        for index, row in relevant_data.iterrows():
            preds_set = error_analysis.get(row['true'], np.array(0))
            preds_set = np.append(preds_set, row['predict'])
            error_analysis[row['true']] = preds_set

        print(f'total acc: {acc}\n')
        print(f'ir/relevant: acc-{relevant_acc}, precision-{precision}, recall-{recall}, f1-{f1}\n')
        print(f'rmse: {rmse}\n')
        print('error_analysis: \n')

        for k, s in error_analysis.items():
            true = np.ones(s.shape)*k
            rmse = mean_squared_error(true, s, squared=False)

            print(f'{str(k)}:    {str(rmse)}')

        return acc, rmse, error_analysis
    

    def run_eval(self) -> None:
        self.eval(self.df.text.values, self.df.label.values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction/Evaluation.')
    
    parser.add_argument('topic', type=str, choices=['gun', 'china', 'abortion', 'drug', 'climate', 'sexual'])
    parser.add_argument('--model', type=str, choices=['guanaco', 'alpaca', 'alpaca-13b', 'falcon'])
    parser.add_argument('--eight_bit', type=bool, default=True)

    config = parser.parse_args()

    evaluator = ModelEval(topic=config.topic, model_abbr=config.model, eight_bit=config.eight_bit)
    evaluator.run_eval()