from tqdm import tqdm
import importlib
import json
import torch
import os
from agents.multi_agent_system import MultiAgentSystem
from agents.base_agent import Agent
from mydatasets.base_dataset import BaseDataset
import pdb

class MDocAgent(MultiAgentSystem):
    def __init__(self, config):
        super().__init__(config)
    
    def first_latex_page_save(self,texts,images):
        general_agent = self.agents[-1]
        # image_agent = self.agents[0]
       
        ###generate corresponding latex response 
        latex_response, latex_messages=general_agent.predict_structure_text(texts,images)

        return latex_response
    
    def attention_analysis_no_text(self,question=None,texts=None,images=None):
        general_agent = self.agents[-1]
        
        full_attns, image_attns,text_attns,last_attns=general_agent.attn_analysis_no_text(question,texts,images)


        return full_attns, image_attns,text_attns,last_attns

    def attention_analysis(self,question=None,texts=None,images=None):
        general_agent = self.agents[-1]
        
        full_attns, image_attns,text_attns,last_attns=general_agent.attn_analysis(question,texts,images)


        return full_attns, image_attns,text_attns,last_attns
    
    
    
    ###use the input text(OCR or structured text) and image to predict
    def ocr_agents_test(self,question,texts,images):
        print(f"This is the entrance of ocr_agents_tests")

        general_agent = self.agents[-1]

        if images and texts:
            print(f"This is general:{texts}")

        try:
            general_response, general_messages = general_agent.predict(question, texts, images, with_sys_prompt=False)
        except ValueError as e:
            print(f"Error occurred: {e}")
            #can add other operation
            return None,None,None

        general_all_messages = "General Agent Answer:\n" + general_response + "\n"
        # image_all_messages = "Image Agent:\n" + image_response + "\n"

        final_general_ans, final_general_messages = self.sum(question+"\n"+general_all_messages)
        # final_image_ans, final_image_messages = self.sum(image_all_messages)
        return final_general_ans, None,None
    
   
    ###原方法---使用image+text进行predict
    def ocr_agents_test_general(self,question,texts,images):

        general_agent = self.agents[-1]


        ###with_sys_promt needed to be set to False
        general_response, general_messages = general_agent.predict(question, texts, images, with_sys_prompt=False)
        # image_response, image_messages = image_agent.predict(question, texts=None, images=images, with_sys_prompt=True)

        general_all_messages = "General Agent Answer:\n" + general_response + "\n"
        # image_all_messages = "Image Agent:\n" + image_response + "\n"

        final_general_ans, final_general_messages = self.sum(question+"\n"+general_all_messages)
        # final_image_ans, final_image_messages = self.sum(image_all_messages)
   
        return final_general_ans, None

    ###based only on images to generate answer
    def images_docqa(self,question,texts,images):
        print(f"This is the entrance of the image_only where texts is None")
        image_agent = self.agents[0]

        try:
            image_response, image_messages = image_agent.predict(question, texts=None, images=images, with_sys_prompt=False)
            print(f"This is the image response: {image_response}")
        except ValueError as e:
            print(f"Error occurred: {e}")
            # can add other operation
            return None,None
        
        image_all_messages = "Image Agent Answer:\n" + image_response + "\n"

        ###use same model to sumamry for evaluation
        final_image_ans, final_image_messages = self.sum(question+"\n"+image_all_messages)
        
        return None, final_image_ans
       


    def test_agents_predict(self,question,texts,images):
        ####can use llavacot for prediction
        llavacot_agent=self.agents[-1]
        qwen25vl_agent = self.agents[0]

        
        ####use same model for summary
        llavacot_response, llavacot_messages=llavacot_agent.predict(question,texts,images,with_sys_prompt=True)
        # qwen25vl_response, qwen25vl_messages=qwen25vl_agent.predict(question,texts,images,with_sys_prompt=True)

        print(f'###llavacot agent: '+llavacot_response)

    

        return llavacot_response, llavacot_response
    
    def ocr_agents_test_structure(self,question,texts,images):
        print(f"This is the entrance of ocr_agents_test_structure")
        general_agent = self.agents[-1]
        ###
        latex_response_ls=[]
        print(f'This is the length of images and texts: {len(images)}')
        print(f'This is the length of images and texts: {len(texts)}')
        if len(texts)>=1 and len(images)>=1:
            for i in range(len(texts)):
                latex_response, latex_messages=general_agent.predict_structure_text([texts[i]],[images[i]])
                print(f'This is the latex response: {latex_response}')
                latex_response_ls.append(latex_response)
            try:
                general_response, general_messages = general_agent.predict(question, latex_response_ls, images, with_sys_prompt=False)
            except ValueError as e:
                print(f"Error occurred: {e}")
            
                return None,None,None
        else:
            try:
                general_response, general_messages = general_agent.predict(question, texts, images, with_sys_prompt=False)
                print(f'This is the general response: {general_response}')
            except ValueError as e:
                print(f"Error occurred: {e}")
                # can add other operation
                return None,None,None
        
        general_all_messages = "General Agent Answer:\n" + general_response + "\n"
      
        final_general_ans, final_general_messages = self.sum(question+"\n"+general_all_messages)
        return final_general_ans, None,None
    

    def ablation_study(self,question,texts):
        print("This is the ablation study")

        general_agent = self.agents[-1]
        # image_agent = self.agents[0]
        
        try:
            general_response = general_agent.ablation_predict(question, texts,0)
        except ValueError as e:
            print(f"Error occurred: {e}")
          
            return None,None,None
        if general_response is None:
            return None,None,None
        
        general_all_messages = "General Agent Answer:\n" + general_response + "\n"
       

        final_general_ans = general_agent.ablation_predict(question+"\n"+general_all_messages,None,1)

   
        return final_general_ans, None,None
    
    

