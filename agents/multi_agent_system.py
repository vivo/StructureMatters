from agents.base_agent import Agent
from mydatasets.base_dataset import BaseDataset
from tqdm import tqdm
import importlib
import json
import torch
from typing import List
import os
import pdb

class MultiAgentSystem:
    def __init__(self, config):
        self.config = config
        self.agents:List[Agent] = []
        self.models:dict = {}
        for agent_config in self.config.agents:
            if agent_config.model.class_name not in self.models:
                module = importlib.import_module(agent_config.model.module_name)
                model_class = getattr(module, agent_config.model.class_name)
                print("Create model: ", agent_config.model.class_name)
                self.models[agent_config.model.class_name] = model_class(agent_config.model)
            self.add_agent(agent_config, self.models[agent_config.model.class_name])
        # pdb.set_trace()
        if config.sum_agent.model.class_name not in self.models:
            module = importlib.import_module(config.sum_agent.model.module_name)
            model_class = getattr(module, config.sum_agent.model.class_name)
            self.models[config.sum_agent.model.class_name] = model_class(config.sum_agent.model)
        self.sum_agent = Agent(config.sum_agent, self.models[config.sum_agent.model.class_name])
        
    def add_agent(self, agent_config, model):
        module = importlib.import_module(agent_config.agent.module_name)
        agent_class = getattr(module, agent_config.agent.class_name)
        agent:Agent = agent_class(agent_config, model)
        self.agents.append(agent)
        
    def predict(self, question, texts, images):
        '''Implement the method in the subclass'''
        pass
    
    def sum(self, sum_question):
        ans, all_messages = self.sum_agent.predict(sum_question)
        def extract_final_answer(agent_response):
            try:
                response_dict = json.loads(agent_response)
                answer = response_dict.get("Answer", None)
                return answer
            except:
                return agent_response
        final_ans = extract_final_answer(ans)
        return final_ans, all_messages
    

        ##draw heat_map for each sample
    def draw_heat_map(self,full_attentions,image_attentions,text_attentions,last_attentions,index,images):
        #output_folder='./heat_map_analysis_57_44_image_overlay_test'
        output_folder=''
        os.makedirs(output_folder, exist_ok=True)
        full_attn_map_dir = os.path.join(output_folder, "full_attn"+str(index), "map")
        
        os.makedirs(full_attn_map_dir, exist_ok=True)

        for i in range(28):
            # attention heatmap
            plt.figure(figsize=(6, 6), dpi=96)
            plt.imshow(full_attentions[i].cpu().float().numpy().reshape(57, 44), cmap='viridis')
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            # 无需额外的title等，直接等比例保存
            plt.savefig(os.path.join(full_attn_map_dir, f"layer_{i}_origin_size.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

        image_attn_map_dir = os.path.join(output_folder, "image_attn"+str(index), "map")
        os.makedirs(image_attn_map_dir, exist_ok=True)
        for i in range(28):
            # attention heatmap
            plt.figure(figsize=(6, 6), dpi=96)
            plt.imshow(image_attentions[i].cpu().float().numpy().reshape(57, 44), cmap='viridis')
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            plt.savefig(os.path.join(image_attn_map_dir, f"layer_{i}_origin_size.png"), bbox_inches='tight', pad_inches=0)
            plt.close()

        if text_attentions:
            text_attn_map_dir = os.path.join(output_folder, "text_attn"+str(index), "map")
            os.makedirs(text_attn_map_dir, exist_ok=True)
            for i in range(28):
                # attention heatmap
                plt.figure(figsize=(6, 6), dpi=96)
                plt.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                plt.savefig(os.path.join(text_attn_map_dir, f"layer_{i}_origin_size.png"), bbox_inches='tight', pad_inches=0)
                plt.close()
        return

    def draw_attention_distribution(self,full_attentions,image_attentions,text_attentions,last_attentions):
        ###ouput path to save figures
        output_folder=''
        images_token_num=2508

        # Save attentions
        os.makedirs(output_folder, exist_ok=True)

        full_attn_dist_dir = os.path.join(output_folder, "full_attn", "dist")
        os.makedirs(full_attn_dist_dir, exist_ok=True)
        full_attn_map_dir = os.path.join(output_folder, "full_attn", "map")
        os.makedirs(full_attn_map_dir, exist_ok=True)
        for i in range(28):
            # attention distribution
            plt.figure(figsize=(6, 6), dpi=96)
            plt.bar(range(1, images_token_num+1), full_attentions[i].numpy())

            plt.title(f"Full Attention Distribution Layer {i+1}", fontsize=16)
            plt.xlabel("Token Index", fontsize=14)
            plt.ylabel("Attention Score", fontsize=14)

            plt.savefig(os.path.join(full_attn_dist_dir, f"layer_{i}.png"))
            plt.close()

            # attention heatmap
            plt.figure(figsize=(6, 6), dpi=96)
            plt.imshow(full_attentions[i].numpy().reshape(57, 44), cmap='viridis')
            plt.axis('off')

            plt.title('Full Attention Map Layer {}'.format(i+1), fontsize=16)
            plt.savefig(os.path.join(full_attn_map_dir, f"layer_{i}.png"))
            plt.close()
        if image_attentions:
            image_attn_dist_dir = os.path.join(output_folder, "image_attn", "dist")
            os.makedirs(image_attn_dist_dir, exist_ok=True)
            image_attn_map_dir = os.path.join(output_folder, "image_attn", "map")
            os.makedirs(image_attn_map_dir, exist_ok=True)
            for i in range(28):
                # attention distribution
                plt.figure(figsize=(6, 6), dpi=96)
                plt.bar(range(1, 2509), image_attentions[i].numpy())

                plt.title(f"Image Attention Distribution Layer {i+1}", fontsize=16)
                plt.xlabel("Token Index", fontsize=14)
                plt.ylabel("Attention Score", fontsize=14)

                plt.savefig(os.path.join(image_attn_dist_dir, f"layer_{i}.png"))
                plt.close()

                # attention heatmap
                plt.figure(figsize=(6, 6), dpi=96)
                plt.imshow(image_attentions[i].numpy().reshape(57, 44), cmap='viridis')
                plt.axis('off')

                plt.title('Image Attention Map Layer {}'.format(i+1), fontsize=16)
                plt.savefig(os.path.join(image_attn_map_dir, f"layer_{i}.png"))
                plt.close()
        
        if text_attentions:
            text_attn_dist_dir = os.path.join(output_folder, "text_attn", "dist")
            os.makedirs(text_attn_dist_dir, exist_ok=True)
            text_attn_map_dir = os.path.join(output_folder, "text_attn", "map")
            os.makedirs(text_attn_map_dir, exist_ok=True)
            for i in range(28):
                # attention distribution
                plt.figure(figsize=(6, 6), dpi=96)
                plt.bar(range(1, 2509), text_attentions[i].numpy())

                plt.title(f"Text Attention Distribution Layer {i+1}", fontsize=16)
                plt.xlabel("Token Index", fontsize=14)
                plt.ylabel("Attention Score", fontsize=14)

                plt.savefig(os.path.join(text_attn_dist_dir, f"layer_{i}.png"))
                plt.close()

                # attention heatmap
                plt.figure(figsize=(6, 6), dpi=96)
                plt.imshow(text_attentions[i].numpy().reshape(57, 44), cmap='viridis')
                plt.axis('off')

                plt.title('Text Attention Map Layer {}'.format(i+1), fontsize=16)
                plt.savefig(os.path.join(text_attn_map_dir, f"layer_{i}.png"))
                plt.close()

        if last_attentions:
            last_attn_dist_dir = os.path.join(output_folder, "last_attn", "dist")
            os.makedirs(last_attn_dist_dir, exist_ok=True)
            last_attn_map_dir = os.path.join(output_folder, "last_attn", "map")
            os.makedirs(last_attn_map_dir, exist_ok=True)
            for i in range(28):
                # attention distribution
                plt.figure(figsize=(6, 6), dpi=96)
                plt.bar(range(1, 2509), last_attentions[i].numpy())

                plt.title(f"Last Attention Distribution Layer {i+1}", fontsize=16)
                plt.xlabel("Token Index", fontsize=14)
                plt.ylabel("Attention Score", fontsize=14)

                plt.savefig(os.path.join(last_attn_dist_dir, f"layer_{i}.png"))
                plt.close()

                # attention heatmap
                plt.figure(figsize=(6, 6), dpi=96)
                plt.imshow(last_attentions[i].numpy().reshape(57, 44), cmap='viridis')
                plt.axis('off')

                plt.title('Last Attention Map Layer {}'.format(i+1), fontsize=16)
                plt.savefig(os.path.join(last_attn_map_dir, f"layer_{i}.png"))
                plt.close()
        return 


    def attention_analysis_structure_dataset(self, dataset:BaseDataset):
        ###the output path of attention results
        attention_output=''
        ###the path with structured text
        latex_path=''

        with open(latex_path,'r') as f:
            latex_response=json.load(f)

        if not os.path.exists(os.path.join(attention_output,'full_attentions.pth')):
            samples = dataset.load_data(use_retreival=True)
            # ###sample images with same size for attetnion analysis(Optional)
            # image_path=''
            # with open(image_path,'r') as f:
            #     questions_maps=json.load(f)

            # ###sample questions 
            # samples_analysis=[]
            # for sa in samples:
            #     if sa['question'] in questions_maps:
            #         samples_analysis.append(sa)
            # print(f'This is the amount of images needed to analysis:{len(samples_analysis)}')
            # samples=samples_analysis
            # ####---------------------------------------------------------------------------------
            samples_no=0
            sample_OOM=0
        
            images_token_num=2508
            ###find the attetnion score in different location
            full_attentions = [torch.zeros(images_token_num, device='cuda') for _ in range(28)]
            image_attentions = [torch.zeros(images_token_num, device='cuda') for _ in range(28)]
            text_attentions = [torch.zeros(images_token_num, device='cuda') for _ in range(28)]
            last_attentions = [torch.zeros(images_token_num, device='cuda') for _ in range(28)]
            for sample in tqdm(samples): 

                if len(eval(sample["evidence_pages"]))>1:
                    continue
                
                question, texts, images = dataset.load_sample_ground_truth_retrieval_data_for_attention_analysis(sample)
                if not images:
                    continue
                ###test single sample to obtain the attention_score
            
                latex_texts=latex_response[question]

                try:
                    full_attns, image_attns,text_attns,last_attns=self.attention_analysis(question,latex_texts,images)
                #####------------------------------------------
                except RuntimeError as e:
                    print(e)
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache() 
                        print(f"This sample cannot be selected due to an OOM error.")
                        print(question)
                        full_attns=None
                        sample_OOM+=1

                if not full_attns:

                    continue


                full_attentions = [full_attns[i] + full_attentions[i] for i in range(28)]
                image_attentions = [image_attns[i] + image_attentions[i] for i in range(28)]
                text_attentions = [text_attns[i] + text_attentions[i] for i in range(28)]
                last_attentions = [last_attns[i] + last_attentions[i] for i in range(28)]
                samples_no+=1

            
            print(f"This is the num of question: {samples_no}")
            print(f"This is the num of OOM error: {sample_OOM}")
                
            ###calculate the final attention scores
            full_attentions = [attn.cpu() / samples_no for attn in full_attentions]
            image_attentions = [attn.cpu() / samples_no for attn in image_attentions]
            text_attentions = [attn.cpu() / samples_no for attn in text_attentions]
            last_attentions = [attn.cpu() / samples_no for attn in last_attentions]

            ###save all attention results of whole dataset
            torch.save(full_attentions,os.path.join(attention_output, 'full_attentions.pth'))
            torch.save(image_attentions,os.path.join(attention_output, 'image_attentions.pth'))
            torch.save(text_attentions,os.path.join(attention_output, 'text_attentions.pth'))
            torch.save(last_attentions,os.path.join(attention_output, 'last_attentions.pth'))



        else:
        ###read the saved attention results
            full_attentions=torch.load(os.path.join(attention_output, 'full_attentions.pth'))
            image_attentions=torch.load(os.path.join(attention_output, 'image_attentions.pth'))
            text_attentions=torch.load(os.path.join(attention_output, 'text_attentions.pth'))
            last_attentions=torch.load(os.path.join(attention_output, 'last_attentions.pth'))

        ####calculate the final score for attention analysis
        self.draw_attention_distribution(full_attentions,image_attentions,text_attentions,last_attentions)

        return full_attentions,image_attentions,text_attentions,last_attentions
    


    def attention_heat_map_image_dataset(self, dataset:BaseDataset):
        samples = dataset.load_data(use_retreival=True)
        ###choose the samples with same size 
        images_path='./results/MMLongBench/images_question_for_heat_map.json'
        with open(images_path,'r') as f:
            questions_maps=json.load(f)

        ####---------------------------------------------------------------------------------
        samples_analysis=[]
        for sa in samples:
            if sa['question'] in questions_maps:
                samples_analysis.append(sa)
        print(f'This is the length of samples needed to analyze attention:{len(samples_analysis)}')
        samples=samples_analysis
        ####---------------------------------------------------------------------------------
        sample_OOM=0
        ###images_token_num=2508
        i=0
        for sample in tqdm(samples): 
            if len(eval(sample["evidence_pages"]))>1:
                continue
            question, texts, images = dataset.load_sample_ground_truth_retrieval_data_for_attention_analysis(sample)
            if not images:
                continue
            texts=None
            try:
                full_attns, image_attns,text_attns,last_attns=self.attention_analysis_no_text(question,texts,images)
            #####------------------------------------------
            except RuntimeError as e:
                print(e)
                if "out of memory" in str(e):
                    torch.cuda.empty_cache() 
                    print(f"This sample cannot be selected due to an OOM error.")
                    print(question)
                    full_attns=None
                    sample_OOM+=1

            ###由于图片过小而返回None值，不做计数

            if not full_attns:
                # image_not_2508+=1
                # image_not_2508_question.append(question)
                continue
            
            ####check whether the attention score of this sample is finished
            ####obtain the attention score
            ###save result for a sample[i]
            # heat_map_output=''
            # os.makedirs(os.path.join(heat_map_output, 'samples'+str(i)), exist_ok=True)
            # torch.save(full_attns,os.path.join(heat_map_output, 'samples'+str(i),'full_attentions.pth'))  
            self.draw_heat_map(full_attns, image_attns,text_attns,last_attns,i,images)
            i+=1
        return
    


    def attention_heat_map_structure_dataset(self, dataset:BaseDataset,latex_path):
        ###use qwen25vl to analyze attention---need to find images with same size 
        samples = dataset.load_data(use_retreival=True)
        if latex_path is None:
            print(f'You need to set the latex path for attention analysis')
            return 
        with open(latex_path,'r') as f:
            latex_response=json.load(f)
            
        # latex_path='./results/MMLongBench/latex_res/latex_response.json'
        ###images_path with same size--you should generate structured text for these samples
        images_path='./results/MMLongBench/images_question_for_heat_map.json'
        with open(images_path,'r') as f:
            questions_maps=json.load(f)

        ####sample--------------------------------------------------------------------------
        samples_analysis=[]
        for sa in samples:
            if sa['question'] in questions_maps:
                samples_analysis.append(sa)
        print(f'This is the length of samples needed to analyze attention:{len(samples_analysis)}')
        samples=samples_analysis
        ####---------------------------------------------------------------------------------
        sample_OOM=0
        ########
        ###images_token_num=2508
        i=0
        for sample in tqdm(samples): 
            if len(eval(sample["evidence_pages"]))>1:
                continue
            question, texts, images = dataset.load_sample_ground_truth_retrieval_data_for_attention_analysis(sample)
            if not images:
                continue
            ###先对每一个样本进行attention_scores测试
            latex_texts=latex_response[question]
            try:
                full_attns, image_attns,text_attns,last_attns=self.attention_analysis(question,latex_texts,images)
            #####------------------------------------------
            except RuntimeError as e:
                print(e)
                if "out of memory" in str(e):
                    torch.cuda.empty_cache() 
                    print(f"This sample cannot be selected due to an OOM error.")
                    print(question)
                    full_attns=None
                    sample_OOM+=1

            if not full_attns:
                continue
            
            ####obtain the attention score
            ###save result for a sample[i]
            # heat_map_output=''
            # os.makedirs(os.path.join(heat_map_output, 'samples'+str(i)), exist_ok=True)
            # torch.save(full_attns,os.path.join(heat_map_output, 'samples'+str(i),'full_attentions.pth'))
            
            self.draw_heat_map(full_attns, image_attns,text_attns,last_attns,i,images)
            i+=1
        
        return 

   

    def attention_analysis_dataset(self, dataset:BaseDataset):
        ###output path to save figures
        attention_output=''
        if not os.path.exists(os.path.join(attention_output,'full_attentions.pth')):

            samples = dataset.load_data(use_retreival=True)
            
            samples_no=0

            ########
            images_token_num=2508
            ###查看不同位置对于images_token的注意力得分
            full_attentions = [torch.zeros(images_token_num, device='cuda') for _ in range(28)]
            image_attentions = [torch.zeros(images_token_num, device='cuda') for _ in range(28)]
            text_attentions = [torch.zeros(images_token_num, device='cuda') for _ in range(28)]
            last_attentions = [torch.zeros(images_token_num, device='cuda') for _ in range(28)]
            for sample in tqdm(samples): 

                if len(eval(sample["evidence_pages"]))>1:
                    continue
                
                question, texts, images = dataset.load_sample_ground_truth_retrieval_data_for_attention_analysis(sample)
                if not images:
                    continue

                try:
                    texts=None
                    full_attns, image_attns,text_attns,last_attns=self.attention_analysis(question,texts,images)
                #####------------------------------------------
                except RuntimeError as e:
                    print(e)
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache() 
                        print(f"This sample cannot be selected due to an OOM error.")
                        print(question)
                        full_attns=None
                        
            
                if not full_attns:
                    continue


                full_attentions = [full_attns[i] + full_attentions[i] for i in range(28)]
                image_attentions = [image_attns[i] + image_attentions[i] for i in range(28)]
                text_attentions = [text_attns[i] + text_attentions[i] for i in range(28)]
                last_attentions = [last_attns[i] + last_attentions[i] for i in range(28)]
                samples_no+=1

            
            print(f"This is the num of question: {samples_no}")
                
            ###计算最终得分
            full_attentions = [attn.cpu() / samples_no for attn in full_attentions]
            image_attentions = [attn.cpu() / samples_no for attn in image_attentions]
            text_attentions = [attn.cpu() / samples_no for attn in text_attentions]
            last_attentions = [attn.cpu() / samples_no for attn in last_attentions]

            ###保存整个数据集的所有结果
            torch.save(full_attentions,os.path.join(attention_output, 'full_attentions.pth'))
            torch.save(image_attentions,os.path.join(attention_output, 'image_attentions.pth'))
            torch.save(text_attentions,os.path.join(attention_output, 'text_attentions.pth'))
            torch.save(last_attentions,os.path.join(attention_output, 'last_attentions.pth'))


        else:
        ###load attention scores to draw
            full_attentions=torch.load(os.path.join(attention_output, 'full_attentions.pth'))
            image_attentions=torch.load(os.path.join(attention_output, 'image_attentions.pth'))
            text_attentions=torch.load(os.path.join(attention_output, 'text_attentions.pth'))
            last_attentions=torch.load(os.path.join(attention_output, 'last_attentions.pth'))

        ####plot
        self.draw_attention_distribution(full_attentions,image_attentions,text_attentions,last_attentions)

        return full_attentions,image_attentions,text_attentions,last_attentions
    

    
    def latex_first_page_save_dataset(self, dataset:BaseDataset):
        latex_res_output='/workspace/mdocagent/results/MMLongBench/latex_res'

        samples = dataset.load_data(use_retreival=True)
        
        print(f'This is the len(samples)==1073 needed to analysis:{len(samples)}')
        latex_res={}
        for sample in tqdm(samples): 
            if len(eval(sample["evidence_pages"]))>1:
                continue
            question, texts, images = dataset.load_sample_ground_truth_retrieval_data_for_attention_analysis(sample)
            if not images:
                continue
            ###先对每一个样本进行attention_scores测试
            latex_response=self.attention_analysis(question,texts,images)
            #####------------------------------------------
            latex_res[question]=latex_response
        
        with open(os.path.join(latex_res_output, 'latex_response.json'), 'w') as f:
            json.dump(latex_res, f)

        return 
    

    
    def predict_dataset_structure(self, dataset:BaseDataset, resume_path = None, data_type='closed-domain',flag=0, latex_path=None):
        ### flag = 1 ---- use predefined LaTeX  
        ### flag = 0 ---- need models to transfer pure OCR results into LaTeX (defaulted)
        samples = dataset.load_data(use_retreival=True)

        if latex_path:
            print(f'This is the latex path :{latex_path}')
            with open(latex_path, 'r') as file:
                latex_path_texts = json.load(file)

        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, 'r') as f:
                samples = json.load(f)
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]


        print(f'This is the length of samoles needed to analysis:{len(samples)}')
        sample_no = 0
        sample_out_of_memory=0
        answer_none=0

   
        for sample in tqdm(samples):
            if resume_path and self.config.ans_key in sample:
                continue
            if data_type=='open-domain':
                question, texts, images= dataset.load_sample_retrieval_data_structure(sample)
            else:
                question, texts, images = dataset.load_sample_ground_truth_retrieval_data(sample)
            if len(images)>10:
                general_agent_ans="not answerable"
            else:
                try:
                    if flag:
                        latex_texts=latex_path_texts[question]
                        if isinstance(latex_texts,str):
                            general_agent_ans, image_agent_ans, messages = self.ocr_agents_test(question, [latex_texts], images)
                        else:
                            print('This is a list')
                            general_agent_ans, image_agent_ans, messages = self.ocr_agents_test(question, latex_texts, images)
                    else:
                        general_agent_ans, image_agent_ans, latex_response = self.ocr_agents_test_structure(question, texts, images)
  
                    print(f'This is the length of image_ans: {image_agent_ans}')
                    if general_agent_ans:
                        if isinstance(general_agent_ans,str):
                            print(f'This is the length of general_ans: {len(general_agent_ans)}')
                        else:
                            general_agent_ans=str(general_agent_ans)
                    else:
                        answer_none+=1
                        print(f'This is question:{question}')
                        general_agent_ans="Not answerable"
                    
                except RuntimeError as e:
                    print(e)
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                    general_agent_ans, image_agent_ans = None, None
                    sample_out_of_memory+=1
                #####save results
            sample[self.config.ans_key] = general_agent_ans
            if self.config.save_message:
                sample[self.config.ans_key+"_image"] = image_agent_ans
            torch.cuda.empty_cache()
            self.clean_messages()
            
            sample_no += 1
        ###save results to file
            if sample_no % self.config.save_freq == 0:
                path = dataset.dump_reults(samples)
                print(f"Save {sample_no} results to {path}.")
        path = dataset.dump_reults(samples)
        print(f"Save final results to {path}.")
        ##-------------------------------------------------  -
        print(f'This is the out of memory: {sample_out_of_memory}')
        print(f"This is the none answer: {answer_none}")
        print(f'This is the latex wrong answer: {latex_wrong}')
    

    def predict_dataset(self, dataset:BaseDataset, resume_path = None, data_type='closed-domain'):
        print(f"This is the entrance of image as input")
        samples = dataset.load_data(use_retreival=True)
        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, 'r') as f:
                samples = json.load(f)
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]

        print(f'This is the length of samples: {len(samples)}')

        sample_no = 0
        sample_out_of_memory=0
        answer_none=0
        for sample in tqdm(samples):
            if resume_path and self.config.ans_key in sample:
                continue
            
            if data_type=='closed-domain':
                question, texts, images = dataset.load_sample_ground_truth_retrieval_data(sample)
            else:
                question, texts, images = dataset.load_sample_retrieval_data(sample)
            if len(images)>10:
                image_agent_ans="not answerable"
            else:
                try:
                    general_agent_ans, image_agent_ans = self.images_docqa(question, texts, images)
                    
                    if image_agent_ans:
                        if isinstance(image_agent_ans,str):
                            print(f'This is the length of image_ans: {len(image_agent_ans)}')
                        else:

                            print(f'This is question:{question}')
                            image_agent_ans=str(image_agent_ans)
                    else:
                        answer_none+=1
                        print(f'This is question:{question}')
                        image_agent_ans="Not answerable"
                except RuntimeError as e:
                    print(e)
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                    general_agent_ans, image_agent_ans = None, None
                    sample_out_of_memory+=1
            #####save results 
            sample[self.config.ans_key] = image_agent_ans
            if self.config.save_message:
                sample[self.config.ans_key+"_image"] = image_agent_ans
            torch.cuda.empty_cache()
            self.clean_messages()
            
            sample_no += 1
        ###save results to file
            if sample_no % self.config.save_freq == 0:
                path = dataset.dump_reults(samples)
                print(f"Save {sample_no} results to {path}.")
        path = dataset.dump_reults(samples)
        print(f"Save final results to {path}.")
        ##-------------------------------------------------  -
        print(f'This is the out of memory: {sample_out_of_memory}')
        print(f"This is the none answer: {answer_none}")



    def predict_dataset_general(self, dataset:BaseDataset, resume_path = None, data_type='closed-domain'):
        print(f'This is the entrance of predict_dataset in multi_agent_system_01.py for image text evaluation')
        samples = dataset.load_data(use_retreival=True)

        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, 'r') as f:
                samples = json.load(f)
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]

        
        print(f'This is the length of samples: {len(samples)}')
        sample_no = 0
        sample_out_of_memory=0
        answer_none=0
        latex_wrong=0
        for sample in tqdm(samples):
            if resume_path and self.config.ans_key in sample:
                continue
            if data_type=='closed-domain':
                question, texts, images = dataset.load_sample_ground_truth_retrieval_data(sample)
            else:
                question, texts, images = dataset.load_sample_retrieval_data(sample)
            print(f"This is the texts : {texts}")
            if len(images)>10:
                general_agent_ans="not answerable"
            else:
                try:
                    general_agent_ans, image_agent_ans, latex_response = self.ocr_agents_test(question, texts, images)
                    
                    print(f'This is the length of image_ans: {image_agent_ans}')
                    if general_agent_ans:
                        if isinstance(general_agent_ans,str):

                            print(f'This is the length of general_ans: {len(general_agent_ans)}')
                        else:
                            general_agent_ans=str(general_agent_ans)
                    else:
                        answer_none+=1
                        print(f'This is question:{question}')
                        general_agent_ans="Not answerable"

                    if latex_response:
                        if isinstance(latex_response,str):
                            print(f'This is the length of latex_response: {len(latex_response)}')
                        else:
                            latex_response=str(latex_response)
                    else:
                        latex_wrong+=1
                        print(f"This is the question that latex response is wrong: {question}")
                        latex_response=None
                    
                except RuntimeError as e:
                    print(e)
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                    general_agent_ans, image_agent_ans = None, None
                    sample_out_of_memory+=1
                #####save results 
            sample[self.config.ans_key] = general_agent_ans

            if self.config.save_message:
                sample[self.config.ans_key+"_image"] = image_agent_ans
            torch.cuda.empty_cache()
            self.clean_messages()
            sample_no += 1
        ###save results to file
            if sample_no % self.config.save_freq == 0:
                path = dataset.dump_reults(samples)
                print(f"Save {sample_no} results to {path}.")
        path = dataset.dump_reults(samples)
        print(f"Save final results to {path}.")
        ##-------------------------------------------------  -
        print(f'This is the out of memory: {sample_out_of_memory}')
        print(f"This is the none answer: {answer_none}")
        print(f'This is the latex wrong answer: {latex_wrong}')



    def ablation_predict_dataset(self, dataset:BaseDataset, resume_path = None,structure_flag=0,latex_path=None):
        samples = dataset.load_data(use_retreival=True)
        print(f'This is the latex path :{latex_path}')
        with open(latex_path, 'r') as file:
            latex_path_texts = json.load(file)

        if resume_path:
            assert os.path.exists(resume_path)
            with open(resume_path, 'r') as f:
                samples = json.load(f)
        if self.config.truncate_len:
            samples = samples[:self.config.truncate_len]

        print(f'This is the length of samples needed to analysis:{len(samples)}')
        sample_no = 0
        sample_out_of_memory=0
        answer_none=0
        latex_wrong=0
        for index,sample in enumerate(tqdm(samples[2110:])):
            if resume_path and self.config.ans_key in sample:
                continue
            #question, texts, images = dataset.load_sample_retrieval_data_structure(sample)
            question, texts, images = dataset.load_sample_ground_truth_retrieval_data(sample)
            if structure_flag:
                print(f"This is the step to transform texts into latex_texts")
                texts=latex_path_texts[question]

            if len(images)>10:
                general_agent_ans="not answerable"
            else:
                try:
                    if isinstance(texts,str):
                        general_agent_ans, image_agent_ans, latex_response = self.ablation_study(question, [texts])
                    else:
                        print('This is list')
                        general_agent_ans, image_agent_ans, latex_response = self.ablation_study(question, texts)

                    print(f'This is the length of image_ans: {image_agent_ans}')
                    if general_agent_ans:
                        if isinstance(general_agent_ans,str):

                            print(f'This is the length of general_ans: {len(general_agent_ans)}')
                        else:
                            general_agent_ans=str(general_agent_ans)
                    else:
                        answer_none+=1
                        print(f'This is question:{question}')
                        general_agent_ans="Not answerable"

                    if latex_response:
                        if isinstance(latex_response,str):
                            print(f'This is the length of latex_response: {len(latex_response)}')
                        else:
                            latex_response=str(latex_response)
                    else:
                        latex_wrong+=1
                        print(f"This is the question that latex response is wrong: {question}")
                        latex_response=None
                    
                except RuntimeError as e:
                    print(e)
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                    general_agent_ans, image_agent_ans = None, None
                    sample_out_of_memory+=1
                #####save results
            sample[self.config.ans_key] = general_agent_ans
           
            if self.config.save_message:
                sample[self.config.ans_key+"_image"] = image_agent_ans
            torch.cuda.empty_cache()
            self.clean_messages()
            
            sample_no += 1
        ###save resaults to file
            if sample_no % self.config.save_freq == 0:
                path = dataset.dump_reults(samples)
                print(f"Save {sample_no} results to {path}.")
        path = dataset.dump_reults(samples)
        print(f"Save final results to {path}.")
        ##-------------------------------------------------  -
        print(f'This is the out of memory: {sample_out_of_memory}')
        print(f"This is the none answer: {answer_none}")
        print(f'This is the latex wrong answer: {latex_wrong}')


    
    def clean_messages(self):
        for agent in self.agents:
            agent.clean_messages()
        # self.sum_agent.clean_messages()

