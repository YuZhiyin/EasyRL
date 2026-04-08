from openai import OpenAI
from tqdm import tqdm
from datetime import datetime
import time
import json
import os 
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# GLOBAL_REASONING_ENHANCE_INDEX = 0

def get_tmp_file_path():
    created_time = time.time()
    created_time = datetime.fromtimestamp(created_time).strftime('%Y-%m-%d %H:%M:%S')
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    res_file_path = './tmp-1101/batch_res_' + timestamp + '.jsonl'
    return res_file_path


class LocalRequest:
    def __init__(self, model_path=None, lora_path=None):
        # Initialize base model with LoRA support if lora_path is provided
        enable_lora = lora_path is not None
        self.model = LLM(
            model=model_path,
            enable_lora=enable_lora,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.9,
            max_model_len=4096
        )
        
        # Store LoRA config if provided
        self.lora_request = None
        if lora_path:
            self.lora_request = LoRARequest(
                "lora_adapter",
                1,
                lora_path
            )
            print(f"LoRA adapter loaded from: {lora_path}")
        else:
            print("No LoRA adapter loaded - using base model only")

        self.results_list = []
        # self.global_reasoning_enhance_index = 0
        self.data_lines = []

    # def single_req(self, msg, config, logprobs=False):
    #     sampling_params = SamplingParams(
    #         temperature=config.get('temperature', 0),
    #         max_tokens=config.get('max_tokens', 100),
    #         logprobs=logprobs
    #     )

    #     # Use chat() method instead of generate()
    #     outputs = self.model.chat(
    #         messages=msg,
    #         sampling_params=sampling_params,
    #         lora_request=self.lora_request
    #     )
        
    #     output = outputs[0]
    #     if logprobs:
    #         return output.outputs[0].text, output.outputs[0].logprobs
    #     return output.outputs[0].text

    """
    def batch_req(self, messages_list, config, save=False, save_dir=''):
        sampling_params = SamplingParams(
            temperature=config.get('temperature', 0),
            max_tokens=config.get('max_tokens', 100),
            logprobs=config.get('logprobs', 0),
            seed=config.get('seed', 0)
        )

        # print sampling_params
        print(sampling_params)

        # Ensure each message in the list is properly formatted
        formatted_messages = []
        for messages in messages_list:
            if isinstance(messages, list):
                formatted_messages.append(messages)
            else:
                formatted_messages.append([messages])
                
        # print(formatted_messages)

        outputs = self.model.chat(
            messages=formatted_messages,
            sampling_params=sampling_params,
            lora_request=self.lora_request,
            use_tqdm=True
        )

        res_list = []
        for output in outputs:
            result = {
                "response": output.outputs[0].text,
            }
            if config.get('logprobs'):
                try:
                    # 添加错误处理
                    logprobs = output.outputs[0].logprobs
                    result["logprobs"] = [next(iter(p.values())).logprob for p in logprobs]
                except (AttributeError, TypeError) as e:
                    print(f"Warning: Could not process logprobs: {e}")
                    result["logprobs"] = []
            res_list.append(result)

        self.results_list = res_list

        if save:
            output_path = save_dir if save_dir else get_tmp_file_path()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # with open(output_path, 'w') as file:
            #     for obj in self.results_list:
            #         file.write(json.dumps(obj) + '\n')
            with open(output_path, 'w', encoding='utf-8') as file:
                for obj in self.results_list:
                    file.write(json.dumps(obj, ensure_ascii=False) + '\n')

            print("保存到: ",output_path)

        return res_list
    """
    # def batch_req(self, messages_list, config, save=False, save_dir=''):
    #     from time import time
    #     import json
    #     import os

    #     sampling_params = SamplingParams(
    #         temperature=config.get('temperature', 0),
    #         max_tokens=config.get('max_tokens', 100),
    #         logprobs=config.get('logprobs', 0),
    #         seed=config.get('seed', 0)
    #     )

    #     print("sampling_params:", sampling_params)

    #     # 【调试点1】检查格式化是否正常
    #     formatted_messages = []
    #     for i, messages in enumerate(messages_list):
    #         if isinstance(messages, list):
    #             formatted_messages.append(messages)
    #         else:
    #             formatted_messages.append([messages])
                
    #     print(formatted_messages)
    #     print(f"[DEBUG] Formatted {len(formatted_messages)} messages.")

    #     # 【调试点2】记录推理开始时间
    #     t0 = time()
    #     print("[DEBUG] Starting model.chat() ...")

    #     outputs = self.model.chat(
    #         messages=formatted_messages,
    #         sampling_params=sampling_params,
    #         lora_request=self.lora_request,
    #         use_tqdm=True
    #     )

    #     print(f"[DEBUG] model.chat() completed in {time() - t0:.2f}s. Output count: {len(outputs)}")

    #     # 【调试点3】每条输出处理单独包 try，避免单点失败导致全卡住
    #     res_list = []
    #     for idx, output in enumerate(outputs):
    #         try:
    #             print(f"[DEBUG] Processing output {idx} ...")
    #             response = output.outputs[0].text
    #             result = {"response": response}

    #             if config.get('logprobs'):
    #                 try:
    #                     logprobs = output.outputs[0].logprobs
    #                     result["logprobs"] = [next(iter(p.values())).logprob for p in logprobs]
    #                 except Exception as e:
    #                     print(f"  [WARNING] Failed logprobs for output {idx}: {e}")
    #                     result["logprobs"] = []

    #             res_list.append(result)
    #         except Exception as e:
    #             print(f"[ERROR] Failed to process output {idx}: {e}")
    #             continue  # 不阻塞其他输出处理

    #     self.results_list = res_list

    #     if save:
    #         output_path = save_dir if save_dir else get_tmp_file_path()
    #         os.makedirs(os.path.dirname(output_path), exist_ok=True)

    #         try:
    #             with open(output_path, 'w', encoding='utf-8') as file:
    #                 for obj in self.results_list:
    #                     file.write(json.dumps(obj, ensure_ascii=False) + '\n')
    #             print("保存到: ", output_path)
    #         except Exception as e:
    #             print("[ERROR] Failed to save results:", e)

    #     return res_list
    def batch_req(self, messages_list, config, save=False, save_dir='', batch_size=512):
        from time import time
        import json
        import os
        from tqdm import tqdm

        sampling_params = SamplingParams(
            temperature=config.get('temperature', 0),
            max_tokens=config.get('max_tokens', 100),
            logprobs=config.get('logprobs', 0),
            seed=config.get('seed', 0)
        )

        print("sampling_params:", sampling_params)

        # 【调试点1】检查格式化是否正常
        formatted_messages = []
        for i, messages in enumerate(messages_list):
            if isinstance(messages, list):
                formatted_messages.append(messages)
            else:
                formatted_messages.append([messages])
        print(f"[DEBUG] Formatted {len(formatted_messages)} messages.")

        # 【优化】批处理推理
        t0 = time()
        print(f"[DEBUG] Starting model.chat() in batches of {batch_size} ...")

        res_list = []
        for start in tqdm(range(0, len(formatted_messages), batch_size), desc="Batches"):
            batch = formatted_messages[start:start + batch_size]
            try:
                outputs = self.model.chat(
                    messages=batch,
                    sampling_params=sampling_params,
                    lora_request=self.lora_request,
                    use_tqdm=False  # 批内不重复显示进度
                )
                print("[DEBUG] successfully return from chat", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed model.chat() for batch starting at {start}: {e}")
                continue
            
            print("[DEBUG] START calculate logprobs!!!", flush=True)
            
            # 【调试点3】每条输出处理单独包 try，避免单点失败导致全卡住
            for idx, output in enumerate(outputs):
                try:
                    response = output.outputs[0].text
                    result = {"response": response}

                    if config.get('logprobs'):
                        try:
                            logprobs = output.outputs[0].logprobs
                            result["logprobs"] = [next(iter(p.values())).logprob for p in logprobs]
                        except Exception as e:
                            print(f"  [WARNING] Failed logprobs for output {start + idx}: {e}")
                            result["logprobs"] = []
                        # print("[DEBUG] we calculate logprobs!!!")
                
                    res_list.append(result)
                except Exception as e:
                    print(f"[ERROR] Failed to process output {start + idx}: {e}")
                    continue  # 不阻塞其他输出处理

        print(f"[DEBUG] model.chat() completed in {time() - t0:.2f}s. Output count: {len(res_list)}", flush=True)

        self.results_list = res_list

        if save:
            output_path = save_dir if save_dir else get_tmp_file_path()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                with open(output_path, 'w', encoding='utf-8') as file:
                    for obj in self.results_list:
                        file.write(json.dumps(obj, ensure_ascii=False) + '\n')
                print("保存到: ", output_path, flush=True)
            except Exception as e:
                print("[ERROR] Failed to save results:", e)

        return res_list


