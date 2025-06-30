from models.base_model import BaseModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
import pdb
from vllm import LLM, SamplingParams
from PIL import Image


class Qwen2VL(BaseModel):
    # def __init__(self, config):
    #     super().__init__(config)
    #     max_pixels = 2048*28*28
    #     self.model = Qwen2VLForConditionalGeneration.from_pretrained(
    #         self.config.model_id, torch_dtype="auto", device_map="balanced_low_0"
    #     )
    #     self.processor = AutoProcessor.from_pretrained(self.config.model_id) # , max_pixels=max_pixels
    #     self.create_ask_message = lambda question: {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": question},
    #         ],
    #     }
    #     self.create_ans_message = lambda ans: {
    #         "role": "assistant",
    #         "content": [
    #             {"type": "text", "text": ans},
    #         ],
    #     }
        

    def __init__(self, config):
        super().__init__(config)
        max_pixels = 2048*28*28
        print(f"This is the vllm for inference")
        self.config = config
        ###使用vllm调用模型进行分片inference
        self.model=LLM(model=self.config.model_id,tensor_parallel_size=2)
        
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }

        
    def create_text_message(self, texts, question):
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
        
    def create_image_message(self, images, question):
        content = []
        for image_path in images:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
    
    ###使用原generate获取attention_scores
    @torch.no_grad()
    def attn_analysis_with_structure(self,question,texts,images):
        ######单个数据接收相关信息，并且分析---直接输入的text就是latex的
        ###attention进行分析时，len(images)==1
        if len(images)!=1:
            print(f"length of images is not equal to 1, something is wrong")
            pdb.set_trace()
        self.clean_up()
        ###text需要改动----直接copy模板即可-
        history=None
        messages = self.process_message(question, texts, images, history)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        # input_test=self.processor(
        #     text=[''.join(texts)],
        #     images=None,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # ###仅仅查看images会被分为几个patch
        # inputs_only_images=self.processor(
        #     text=['<|vision_start|><|image_pad|><|vision_end|>'],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # ).to("cuda")
        # # output_ids_images=self.model.generate(**inputs_only_images, max_new_tokens=1,output_attentions=True,output_hidden_states=True,return_dict_in_generate=True)
        # # output_ids_images=self.model.generate(**inputs_only_images, max_new_tokens=1,output_attentions=True,output_hidden_states=True,return_dict_in_generate=True)
        # if inputs_only_images['input_ids'].shape[1]!=2510:
        #     print(f"The num of images_token is not 2510, something is wrong")
        #     pdb.set_trace()
        inputs = inputs.to("cuda")        
        outputs_ids = self.model.generate(**inputs, max_new_tokens=1,output_attentions=True,output_hidden_states=True,return_dict_in_generate=True)
        
        print(f"check the attention scores: {type(outputs_ids)}")


        ####这里需要分清哪个部分是text token 哪个部分是image token 
        ####共有28层 
        print(f"This is the shape of outputs_ids['attentions'][0]==28: {len(outputs_ids['attentions'][0])}")
        images_evidence_token_num=2508
        # texts_evidence_token_num_before_images=input_test['input_ids'].shape[1]
        # texts_images_evidence_num=texts_evidence_token_num_before_images+images_evidence_token_num

        ####直接抓取images token的索引即可
        images_token_indices=torch.nonzero(inputs['input_ids']==151655)
        images_token_start=images_token_indices[0][1]
        images_token_end=images_token_indices[-1][1]
        if (images_token_end-images_token_start+1)!=2508:
            print(f"The num of images_token is not 2508, something is wrong")
            return None,None,None,None

        all_attns = [attns[0, :, 14:,images_token_start:(images_token_end+1)].mean(dim=0) for attns in outputs_ids['attentions'][0]] # (N, 2508) * 28
        print(f"This is the shape of all_attns: {all_attns[0].shape}")

        full_attns = [all_attn.sum(dim=0) / (torch.arange(2508, 0, -1, device='cuda') + all_attn.shape[0] - 2508) for all_attn in all_attns] # (2508,) * 28
        print(f"This is the shape of full_attns: {full_attns[0].shape}")

        ###下面两个就是细粒度的我想要的互相交互的分析。
        ####查看image作为query如何查看text_token-----自己内部如何看自己----可以分析
        image_attns = [all_attn[(images_token_start-14):(images_token_end-14+1), :].sum(dim=0) / torch.arange(2508, 0, -1, device='cuda') for all_attn in all_attns] # (2508,) * 28
        print(f"This is the shape of image_attns: {image_attns[0].shape}")
        ###查看text作为query如何查看image_query------texts部分如何看image_token----可以分析
        text_attns = [all_attn[:(images_token_start-14), :].sum(dim=0) / (images_token_start-14) for all_attn in all_attns] # (2508,) * 28
        print(f"This is the shape of text_attns: {text_attns[0].shape}")

        ###生成时候如何关注images和texts的token----目前设置max_new_tokens=1----后续可能需要调整
        last_attns = [all_attn[-1, :] for all_attn in all_attns] # (2508,) * 28
        print(f"This is the shape of last_attns: {last_attns[0].shape}")
        
        # pdb.set_trace()
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        # output_text = self.processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )[0]
        # messages.append(self.create_ans_message(output_text))
        self.clean_up()

        return full_attns, image_attns,text_attns,last_attns
    
    ###使用原generate获取attention_scores
    @torch.no_grad()
    def attn_analysis(self,question,texts,images):
        ######单个数据接收相关信息，并且分析
        ###attention进行分析时，len(images)==1
        if len(images)!=1:
            print(f"length of images is not equal to 1, something is wrong")
            pdb.set_trace()
        self.clean_up()
        ###text需要改动----直接copy模板即可-
        history=None
        messages = self.process_message(question, texts, images, history)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        # input_test=self.processor(
        #     text=[''.join(texts)],
        #     images=None,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # ###仅仅查看images会被分为几个patch
        # inputs_only_images=self.processor(
        #     text=['<|vision_start|><|image_pad|><|vision_end|>'],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # ).to("cuda")
        # # output_ids_images=self.model.generate(**inputs_only_images, max_new_tokens=1,output_attentions=True,output_hidden_states=True,return_dict_in_generate=True)
        # # output_ids_images=self.model.generate(**inputs_only_images, max_new_tokens=1,output_attentions=True,output_hidden_states=True,return_dict_in_generate=True)
        # if inputs_only_images['input_ids'].shape[1]!=2510:
        #     print(f"The num of images_token is not 2510, something is wrong")
        #     pdb.set_trace()
        inputs = inputs.to("cuda")        
        outputs_ids = self.model.generate(**inputs, max_new_tokens=1,output_attentions=True,output_hidden_states=True,return_dict_in_generate=True)
        
        print(f"check the attention scores: {type(outputs_ids)}")


        ####这里需要分清哪个部分是text token 哪个部分是image token 
        ####共有28层 
        print(f"This is the shape of outputs_ids['attentions'][0]==28: {len(outputs_ids['attentions'][0])}")
        images_evidence_token_num=2508
        # texts_evidence_token_num_before_images=input_test['input_ids'].shape[1]
        # texts_images_evidence_num=texts_evidence_token_num_before_images+images_evidence_token_num

        ####直接抓取images token的索引即可
        images_token_indices=torch.nonzero(inputs['input_ids']==151655)
        images_token_start=images_token_indices[0][1]
        images_token_end=images_token_indices[-1][1]
        if (images_token_end-images_token_start+1)!=2508:
            print(f"The num of images_token is not 2508, something is wrong")
            return None,None,None,None

        all_attns = [attns[0, :, 14:,images_token_start:(images_token_end+1)].mean(dim=0) for attns in outputs_ids['attentions'][0]] # (N, 2508) * 28
        print(f"This is the shape of all_attns: {all_attns[0].shape}")

        full_attns = [all_attn.sum(dim=0) / (torch.arange(2508, 0, -1, device='cuda') + all_attn.shape[0] - 2508) for all_attn in all_attns] # (2508,) * 28
        print(f"This is the shape of full_attns: {full_attns[0].shape}")

        ###下面两个就是细粒度的我想要的互相交互的分析。
        ####查看image作为query如何查看text_token-----自己内部如何看自己----可以分析
        image_attns = [all_attn[(images_token_start-14):(images_token_end-14+1), :].sum(dim=0) / torch.arange(2508, 0, -1, device='cuda') for all_attn in all_attns] # (2508,) * 28
        print(f"This is the shape of image_attns: {image_attns[0].shape}")
        ###查看text作为query如何查看image_query------texts部分如何看image_token----可以分析
        text_attns = [all_attn[:(images_token_start-14), :].sum(dim=0) / (images_token_start-14) for all_attn in all_attns] # (2508,) * 28
        print(f"This is the shape of text_attns: {text_attns[0].shape}")

        ###生成时候如何关注images和texts的token----目前设置max_new_tokens=1----后续可能需要调整
        last_attns = [all_attn[-1, :] for all_attn in all_attns] # (2508,) * 28
        print(f"This is the shape of last_attns: {last_attns[0].shape}")
        
        # pdb.set_trace()
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        # output_text = self.processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )[0]
        # messages.append(self.create_ans_message(output_text))
        self.clean_up()

        return full_attns, image_attns,text_attns,last_attns
    
    @torch.no_grad()
    def predict_structure_text(self,texts=None,images=None):
        ###仍然调用Qwen2.5VL----产生latex结构代码----由于latex结构的代码会产生很多多余符号，因此当前转化时只处理evidences_sources里面的第一页。
        history=None       
        messages = self.process_message(None,texts, images, history)
        texts_info=""
        for txt in texts:
            texts_info+=txt
      
        images_ls=[]
        for img in images:
            images_ls.append(Image.open(img))
        image_holder="<|vision_start|><|image_pad|><|vision_end|>"*len(images_ls)


        text=("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                f"Texts: {texts_info}\n"
                f"Imgaes: {image_holder}\n"
                "You are a professional assistant for converting mixed image-text content into standardized LaTeX documents. Strictly follow these processing rules:\n"
                "**Input Data**:\n"
                "- Texts: Plain text sequences from OCR recognition (no structural information)\n"
                "- Images: Original image array\n"

                "**Processing Workflow**:\n"
                "1. **Content Classification**:\n"
                    "- Analyze each image for:\n"
                        "a) Plain text paragraphs\n"
                        "b) Mathematical formulas\n"
                        "c) Data tables/charts\n"
                        "d) Non-text images\n"
                
                "2. **Conversion Strategies**:\n"
                    "- **Mathematical Formulas**:\n"
                        "* Inline equations: `$...$`\n"
                        "* Standalone equations: `\\begin{equation}...\\end{equation}`\n"
                        "* Aligned equations: `\\begin{align}...\\end{align}`\n"
                    
                    "- **Table Processing**:\n"
                        "* Auto-detect columns to create tabular environments\n"
                        "* Add `\\hline` separators\n"
                        "* Example:\n"
                        "\\begin{tabular}{|c|c|}\n"
                        "\\hline\n"
                        "Header1 & Header2 \\\\\n"
                        "\\hline\n"
                        "Data1 & Data2 \\\\\n"
                        "\\hline\n"
                        "\\end{tabular}\n"

                    "- **Image Placeholders**:\n"
                        "* Use `\\includegraphics[width=0.8\\textwidth]{IMAGE_PLACEHOLDER_[index]}`\n"
                        "* Preserve original image index numbering\n"

                "3. **Structure Reconstruction**:\n"
                    "- Recognize document structures:\n"
                        "* Section headers (`\\section{}`)\n"
                        "* Numbered lists (`\\begin{enumerate}`)\n"
                        "* Bullet points (`\\begin{itemize}`)\n"
                        "* Paragraph separation (empty line spacing)\n"

                    "**Output Requirements**:\n"
                        "- Final output must be compilable LaTeX\n"
                        "- Preserve original content order\n"
                        "- Auto-complete document framework\n"
                        "- Use `booktabs` package for table styling\n"
                        "- Generate pure LaTeX code, do not include any code block markup or comments\n"

                    "**Error Handling**:\n"
                        "- Unrecognized formulas: `\\text{[Formula OCR Error]}`\n"
                        "- Broken tables: Keep raw data rows\n"
                        "- Missing images: Maintain placeholder markers\n"

                    "**Formatting Constraints**:\n"
                        "- Output LaTeX content directly, do not wrap it with any markup\n"
                f"<|im_end|>\n<|im_start|>assistant\n")

        
        ###使用vllm加速推理-------------------------------------------------------------------
        sampling_params = SamplingParams(
                        temperature=0.0,
                        top_k=1,
                        top_p=1.0,
                        stop_token_ids=[],
                        repetition_penalty=1.05,
                        max_tokens=8192,
                    )

        if images:
            inputs= {
                "prompt":text,
                "multi_modal_data": {
                    "image":images_ls
                }
            }
        else:
            inputs= {
                "prompt":text
            }
        # print(f"------------------This is the device--------------")
        # for k,v in inputs.items():
        #     print(v.device)
        # inputs = {k: v.to("cuda:0") for k, v in inputs.items()} 

        # print(f"------------------This is the after device--------------")
        # for k,v in inputs.items():
        #     print(v.device)

        outputs = self.model.generate([inputs], sampling_params=sampling_params)



        output_text=outputs[0].outputs[0].text
        ###--------------------------------------------------------------------------------------

        messages.append(self.create_ans_message(output_text))
        self.clean_up()
        return output_text, messages

    ####only with text or structured text to predict
    @torch.no_grad()
    def ablation_predict(self, question, texts = None,sum_flag = 0):
        self.clean_up()

        if  texts:
            print(f"This is the predict processing")
            texts_info=""
            for txt in texts:
                texts_info+=txt
            text=("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                 "<|im_start|>user\n"
                f"Texts: {texts_info}\n"
                "You are an advanced agent capable of analyzing both text and images. Your task is to use both the textual and visual information provided to answer the user’s question accurately.\n"
                "Extract Text from Both Sources: If the image contains text, extract it using OCR, and consider both the text in the image and the provided textual content.\n"
                "Analyze Visual and Textual Information: Combine details from both the image (e.g., objects, scenes, or patterns) and the text to build a comprehensive understanding of the content.\n"
                "Provide a Combined Answer: Use the relevant details from both the image and the text to provide a clear, accurate, and context-aware response to the user's question.\n"
                "When responding:\n"
                "If both the image and text contain similar or overlapping information, cross-check and use both to ensure consistency.\n"
                "If the image contains information not present in the text, include it in your response if it is relevant to the question.\n"
                "If the text and image offer conflicting details, explain the discrepancies and clarify the most reliable source.\n"
                "Since you have access to both text and image data, you can provide a more comprehensive answer than agents with single-source data.\n"
                f"Question: \n{question}<|im_end|>\n<|im_start|>assistant\n")
        else:
            if sum_flag:
                print("----------------------------------------------------------")
                print(f"This is the summary processing")
                ###此处处理的是sum_agent的prompt
                text=("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n"
                "You are tasked with summarizing and evaluating the collective responses provided by other agent. You have access to the following information:\n"
                "Answers: The answer from other agent.\n"
                "Using this information, perform the following tasks:\n"
                "Analyze: Evaluate the quality, consistency, and relevance of each answer. Identify commonalities, discrepancies, or gaps in reasoning.\n"
                "Synthesize: Summarize the most accurate and reliable information based on the evidence provided by the agents and their discussions.\n"
                "Conclude: Provide a final, well-reasoned answer to the question or task. Your conclusion should reflect the consensus (if one exists) or the most credible and well-supported answer.\n"
                "Based on the origin question and provided answer from other agent, summarize the final decision clearly. You should only return the final answer in this dictionary format: \n"
                '{"Answer": "Your final answer here."}\n'
                "Don't give other information.\n"
                f"Question:{question}<|im_end|>\n<|im_start|>assistant\n")
            else:
                ###在text和image均缺失并且不是需要summarize的情况
                return None
        print(f"This is the texts:{text}")
        
        sampling_params = SamplingParams(
                        max_tokens=8192
                    )
      
        inputs= {
            "prompt":text
        }

        outputs = self.model.generate([inputs], sampling_params=sampling_params)

        output_text=outputs[0].outputs[0].text
        self.clean_up()

        return output_text



    ####并行推理的predict
    @torch.no_grad()
    def predict(self, question, texts = None, images = None, history = None):
        self.clean_up()

        # pdb.set_trace()
        ###不在一个agent系统中，因此不要传入history消息
        history=None       
        messages = self.process_message(question, texts, images, history)
        # text = self.processor.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        images_ls=[]
        if images:
            for img in images:
                images_ls.append(Image.open(img))
        image_holder="<|vision_start|><|image_pad|><|vision_end|>"*len(images_ls)
        ####适用于vLLM推理的text 给出image_agent的处理模板
        if  texts:
            texts_info=""
            for txt in texts:
                texts_info+=txt
            text=("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                 "<|im_start|>user\n"
                f"Texts: {texts_info}\n"
                f"Imgaes: {image_holder}\n"
                "You are an advanced agent capable of analyzing both text and images. Your task is to use both the textual and visual information provided to answer the user’s question accurately.\n"
                "Extract Text from Both Sources: If the image contains text, extract it using OCR, and consider both the text in the image and the provided textual content.\n"
                "Analyze Visual and Textual Information: Combine details from both the image (e.g., objects, scenes, or patterns) and the text to build a comprehensive understanding of the content.\n"
                "Provide a Combined Answer: Use the relevant details from both the image and the text to provide a clear, accurate, and context-aware response to the user's question.\n"
                "When responding:\n"
                "If both the image and text contain similar or overlapping information, cross-check and use both to ensure consistency.\n"
                "If the image contains information not present in the text, include it in your response if it is relevant to the question.\n"
                "If the text and image offer conflicting details, explain the discrepancies and clarify the most reliable source.\n"
                "Since you have access to both text and image data, you can provide a more comprehensive answer than agents with single-source data.\n"
                f"Question: \n{question}<|im_end|>\n<|im_start|>assistant\n")
        else:
            if images:
                text=("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n"
                    f"Imgaes: {image_holder}\n"
                    "You are an advanced image processing agent specialized in analyzing and extracting information from images. The images may include document screenshots, illustrations, or photographs. Your primary tasks include:\n"
                    "Extracting textual information from images using Optical Character Recognition (OCR).\n"
                    "Analyzing visual content to identify relevant details (e.g., objects, patterns, scenes).\n"
                    "Combining textual and visual information to provide an accurate and context-aware answer to user's question.\n"
                    "Remeber you can only get the information from the images provided\n"
                    f"Question: \n{question}<|im_end|>\n<|im_start|>assistant\n")
            else:
                ###此处处理的是sum_agent的prompt
                text=("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n"
                "You are tasked with summarizing and evaluating the collective responses provided by other agent. You have access to the following information:\n"
                "Answers: The answer from other agent.\n"
                "Using this information, perform the following tasks:\n"
                "Analyze: Evaluate the quality, consistency, and relevance of each answer. Identify commonalities, discrepancies, or gaps in reasoning.\n"
                "Synthesize: Summarize the most accurate and reliable information based on the evidence provided by the agents and their discussions.\n"
                "Conclude: Provide a final, well-reasoned answer to the question or task. Your conclusion should reflect the consensus (if one exists) or the most credible and well-supported answer.\n"
                "Based on the origin question and provided answer from other agent, summarize the final decision clearly. You should only return the final answer in this dictionary format: \n"
                '{"Answer": "Your final answer here."}\n'
                "Don't give other information.\n"
                f"Question:{question}<|im_end|>\n<|im_start|>assistant\n")
        sampling_params = SamplingParams(
                        max_tokens=self.config.max_new_tokens
                    )
        # ###image_inputs---应该是PIL对象组成的list
        # image_inputs, video_inputs = process_vision_info(messages)
        
        if images:
            inputs= {
                "prompt":text,
                "multi_modal_data": {
                    "image":images_ls
                }
            }
        else:
            inputs= {
                "prompt":text
            }

        # inputs = self.processor(
        #     text=[text],
        #     images=image_inputs,
        #     videos=video_inputs,
        #     padding=True,
        #     return_tensors="pt",
        # )
        # inputs = inputs.to("cuda")

        outputs = self.model.generate([inputs], sampling_params=sampling_params)

        output_text=outputs[0].outputs[0].text

        # generated_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        # generated_ids_trimmed = [
        #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        # ]
        # output_text = self.processor.batch_decode(
        #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )[0]
        
        messages.append(self.create_ans_message(output_text))
        self.clean_up()
        return output_text, messages
        
    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], list):
                return False
            for content in item["content"]:
                if not isinstance(content, dict):
                    return False
                if "type" not in content:
                    return False
                if content["type"] not in content:
                    return False
        return True

class Qwen2_5VL(Qwen2VL):
    ##vLLM加速后的模型
    def __init__(self, config):
        print(f"This is the vllm for inference")
        self.config = config
        ###使用vllm调用模型进行分片inference
        self.model=LLM(model=self.config.model_id,tensor_parallel_size=1)
        
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }

    ###原始模型load 
    # def __init__(self, config):
    #     self.config = config
    #     self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #         self.config.model_id, torch_dtype="auto", device_map="balanced_low_0"
    #     )
    #     self.processor = AutoProcessor.from_pretrained(self.config.model_id)
    #     self.create_ask_message = lambda question: {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": question},
    #         ],
    #     }
    #     self.create_ans_message = lambda ans: {
    #         "role": "assistant",
    #         "content": [
    #             {"type": "text", "text": ans},
    #         ],
    #     }


    
    