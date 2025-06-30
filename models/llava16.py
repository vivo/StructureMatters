from models.base_model import BaseModel
from transformers import AutoProcessor
import torch
import pdb
from vllm import LLM, SamplingParams
from PIL import Image
from vllm.assets.image import ImageAsset


class Llava16(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        max_pixels = 2048*28*28
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
     
    @torch.no_grad()
    def predict_structure_text(self,texts=None,images=None):
        ###仍然调用Qwen2.5VL----产生latex结构代码----由于latex结构的代码会产生很多多余符号，因此当前转化时只处理evidences_sources里面的第一页。
        history=None       
        messages = self.process_message(None,texts, images, history)
        texts_info=""
        for txt in texts:
            texts_info+=txt

        images_ls=[]
        image_holder=""
        if images:
            for id,img in enumerate(images):
                images_ls.append(Image.open(img))
                tmp='<image>'
                image_holder+=tmp
        print(f"This is the image holder:{image_holder}")


        text=("[INST]"
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
                "[/INST]")

        
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
       

        outputs = self.model.generate([inputs], sampling_params=sampling_params)

        output_text=outputs[0].outputs[0].text
        ###--------------------------------------------------------------------------------------

        messages.append(self.create_ans_message(output_text))
        self.clean_up()
        return output_text, messages




    ####并行推理的predict
    @torch.no_grad()
    def predict(self, question, texts = None, images = None, history = None):
        self.clean_up()

        ###不在一个agent系统中，因此不要传入history消息
        history=None       
        messages = self.process_message(question, texts, images, history)
        # text = self.processor.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        images_ls=[]
        image_holder=""
        if images:
            for id,img in enumerate(images):
                images_ls.append(Image.open(img))
                tmp='<image>'
                image_holder+=tmp
                
        #print(f"This is the image holder:{image_holder}")

        
        ####适用于vLLM推理的text 给出image_agent的处理模板
        if  texts:
            texts_info=""
            for txt in texts:
                texts_info+=txt
            text=("[INST]"
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
                f"Question: \n{question}[/INST]")
        else:
            if images:
                text=("[INST]"
                    f"Images: {image_holder}\n"
                    "You are an advanced image processing agent specialized in analyzing and extracting information from images. The images may include document screenshots, illustrations, or photographs. Your primary tasks include:\n"
                    "Extracting textual information from images using Optical Character Recognition (OCR).\n"
                    "Analyzing visual content to identify relevant details (e.g., objects, patterns, scenes).\n"
                    "Combining textual and visual information to provide an accurate and context-aware answer to user's question.\n"
                    "Remeber you can only get the information from the images provided\n"
                    f"Question: \n{question}[/INST]")
            else:
                ###此处处理的是sum_agent的prompt
                text=("[INST]"
                "You are tasked with summarizing and evaluating the collective responses provided by other agent. You have access to the following information:\n"
                "Answers: The answer from other agent.\n"
                "Using this information, perform the following tasks:\n"
                "Analyze: Evaluate the quality, consistency, and relevance of each answer. Identify commonalities, discrepancies, or gaps in reasoning.\n"
                "Synthesize: Summarize the most accurate and reliable information based on the evidence provided by the agents and their discussions.\n"
                "Conclude: Provide a final, well-reasoned answer to the question or task. Your conclusion should reflect the consensus (if one exists) or the most credible and well-supported answer.\n"
                "Based on the origin question and provided answer from other agent, summarize the final decision clearly. You should only return the final answer in this dictionary format: \n"
                '{"Answer": "Your final answer here."}\n'
                "Don't give other information.\n"
                f"Question:{question}[/INST]")
        
        print(f"This is the text: {text}")
        sampling_params = SamplingParams(
                        max_tokens=2048
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
    
        ####only with text or structured text to predict
    @torch.no_grad()
    def ablation_predict(self, question, texts = None,sum_flag = 0):
        self.clean_up()

        if  texts:
            print(f"This is the predict processing")
            texts_info=""
            for txt in texts:
                texts_info+=txt
            text=("[INST]"
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
                f"Question: \n{question}[/INST]")
        else:
            if sum_flag:
                print("----------------------------------------------------------")
                print(f"This is the summary processing")
                ###此处处理的是sum_agent的prompt
                text=("[INST]"
                "You are tasked with summarizing and evaluating the collective responses provided by other agent. You have access to the following information:\n"
                "Answers: The answer from other agent.\n"
                "Using this information, perform the following tasks:\n"
                "Analyze: Evaluate the quality, consistency, and relevance of each answer. Identify commonalities, discrepancies, or gaps in reasoning.\n"
                "Synthesize: Summarize the most accurate and reliable information based on the evidence provided by the agents and their discussions.\n"
                "Conclude: Provide a final, well-reasoned answer to the question or task. Your conclusion should reflect the consensus (if one exists) or the most credible and well-supported answer.\n"
                "Based on the origin question and provided answer from other agent, summarize the final decision clearly. You should only return the final answer in this dictionary format: \n"
                '{"Answer": "Your final answer here."}\n'
                "Don't give other information.\n"
                f"Question:{question}[/INST]")
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


