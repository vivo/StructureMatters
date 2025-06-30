from models.base_model import BaseModel
from mydatasets.base_dataset import BaseDataset
import os
from typing import Dict, Union
import json
import pandas as pd
from tqdm import tqdm
import re
import importlib
import pdb

class Agent:
    def __init__(self, config, model=None):
        self.config = config
        self.messages = None
        if model is not None:
            self.model:BaseModel = model
        else:
            module = importlib.import_module(self.config.model.module_name)
            model_class = getattr(module, self.config.model.class_name)
            print("Create model: ", self.config.model.class_name)
            self.model = model_class(self.config.model)

    
    def extract_substring_with_quotes(self,input_string, quotes="'''"):
        pattern = f"{quotes}(.*?){quotes}"
        matches = re.findall(pattern, input_string, re.DOTALL)
        for i in range(len(matches)):
            if matches[i][:5] == 'json\n':
                matches[i] = matches[i][5:]
            if matches[i][:6] == 'latex\n':
                matches[i] = matches[i][6:]


        if len(matches) == 1:
            return matches[0]
        else:
            return ''.join(matches)
    

    def try_extract_content_from_quotes(self,content):
        if "'''" in content:
            return self.extract_substring_with_quotes(content)
        elif "```" in content:
            return self.extract_substring_with_quotes(content, quotes="```")
        else:
            return content

    
    def clean_messages(self):
        self.messages = None
    
    def attn_analysis_no_text(self,question=None,texts=None,images=None):
        output=self.model.attn_analysis_no_text(question,texts,images)
        return output

    def attn_analysis(self,question=None,texts=None,images=None):
        output=self.model.attn_analysis(question,texts,images)

        return output

    def ablation_predict(self, question, texts = None,sum_flag=0):
        generated_ans=self.model.ablation_predict(question, texts, sum_flag)
        return generated_ans


    def predict_structure_text(self,texts=None,images=None):

        ###这里处理返回的结果，后处理以便更加符合格式
        generated_ans, messages=self.model.predict_structure_text(texts,images)
        generated_ans=self.try_extract_content_from_quotes(generated_ans)

        return generated_ans,messages
        
    def _predict(self, question, texts=None, images=None, add_to_message = False):
        if not self.config.agent.use_text:
            texts = None
        if not self.config.agent.use_image:
            images = None
        generated_ans, messages = self.model.predict(question, texts, images, self.messages)
        generated_ans=self.try_extract_content_from_quotes(generated_ans)
        if add_to_message:
            self.messages = messages
        return generated_ans, messages
    
    def predict(self, question, texts=None, images=None, with_sys_prompt=False):
        if with_sys_prompt:
            question = self.config.agent.system_prompt + question
        return self._predict(question, texts, images, add_to_message = True)
    
    def self_reflect(self, prompt=None, add_to_message = True):
        if prompt is None:
            self_reflect_prompt = self.config.agent.self_reflect_prompt
        else:
            self_reflect_prompt = prompt
        
        generated_ans, messages = self._predict(question = self_reflect_prompt)
        if add_to_message:
            self.messages = messages
        
        return generated_ans
    
    def eval(self, question, answer, gt):
        prompt = self.config.agent.eval_system_prompt.format(question=question, answer=answer, gt=gt)
        try:
            generated_ans, _ = self.model.predict(prompt)
            result = extract_evaluation_metrics(generated_ans)
            return result
        except Exception as e:
            print(f"Error evaluating answer: {str(e)}")
            return {"binary_correctness": 0}
    
    def eval_dataset(self, dataset: BaseDataset):
        samples, ans_path = dataset.load_latest_results()
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]
        samples_with_answer = []
        for sample in tqdm(samples):
            try:
                question = sample[dataset.config.question_key]
                answer = sample[self.config.ans_key]
                gt = sample[dataset.config.gt_key]
                result = self.eval(question, answer, gt)
                sample['binary_correctness'] = result.get('binary_correctness', None)
                samples_with_answer.append(sample)
            except Exception as e:
                print(f"Error evaluating sample: {str(e)}")
                
        ans_file_path_name = ans_path[:-5]+"_results.json"
        with open(ans_file_path_name, "w") as file:
            json.dump(samples_with_answer, file, indent=4)
            
        samples_with_answer = pd.DataFrame(samples_with_answer)
        path = os.path.join(dataset.config.result_dir,"results.txt")
        with open(path, "a") as file:
            file.write("\nEvaluation Results Summary:\n")
            file.write(f"Result file: {ans_path}\n")
            file.write(f"Average Binary Correctness: {samples_with_answer['binary_correctness'].mean():.3f}\n")
        
        print(f"Save results to {path}.")

        #####analysis bad case and good case 
        print(f'Save good and bad case to analysis')
        path_good_case=ans_path[:-5]+"good_case.json"
        path_bad_case=ans_path[:-5]+"bad_case.json"
        bad_case=[]
        good_case=[]
        for sample in samples_with_answer:
            if not sample['binary_correctness']:
                bad_case.append(sample)
            else:
                good_case.append(sample)
        print(f'This is the count of bad case:{len(bad_case)}')
        print(f'This is the count of good case:{len(good_case)}')

        with open(path_good_case,'w') as file:
            json.dump(good_case,file,indent=4)
        
        with open(path_bad_case,'w') as file:
            json.dump(bad_case,file,indent=4)

def extract_evaluation_metrics(eval_str: str) -> Dict[str, Union[float, int]]:
    try:
        start_index = eval_str.find('{') 
        end_index = eval_str.rfind('}') + 1 
        eval_str = eval_str[start_index:end_index]
        metrics = json.loads(eval_str)
        return {
            'binary_correctness': int(metrics.get('binary_correctness', 0))
        }
    except json.JSONDecodeError as e:
        return {
            'binary_correctness': 0
        }
    except Exception as e:
        return {
            'binary_correctness': 0
        }