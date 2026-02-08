import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Notice Preprocess class Must be named "Model"
class Model(object):
    
    def load_model(self, model_name: str, model_path: str, device: str, dtype: torch.dtype):
        """ `Load and initialize your model`

        Args:
            model_path (str): Local path of the model. Can be path to a folder or a file
            device (str): string of the model's device. It is determined by th inference system: auto, cuda:0, cuda:1, ..., cpu
            dtype (torch.dtype): torch data type that can be set for model's parameter precision in inference.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.to(device)
        self.model_name = model_name

    def preprocess(self, body: any) -> any:
        """ Preprocessing function

        Args:
            body (any): Any type of data may be accepted. Json and bytes data are tested and guaranteed to be processed from the API

        Returns:
            any: preprocessed data, ready to be fed into the model
        """
        if 'messages' in body:
            messages = body['messages']
            model = body['model']
            max_tokens = body["max_tokens"]
        else:
            prompts = body['prompts']
            model = body['model_name']
            max_tokens = body["max_tokens"]
            messages = []
            for prompt in prompts:
                messages.append({
                    'role': prompt['role'],
                    'content': prompt['text']
                })
        if model != self.model_name:
            raise ValueError(f"Input model {model} is not compatible with served model {self.model_name}")
        # messages = json.loads(user_req)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        return {"model_inputs": model_inputs, "max_tokens": max_tokens}

    def process(
            self,
            data: any,
    ) -> dict:
        """ Main processing function

        Args:
            data (any): Any type of data may be accepted. The data is directly the output of the preprocess function.

        Returns:
            dict: A dict type, json serializable output should be returned.
        """
        generated_ids = self.model.generate(
            **(data["model_inputs"]),
            max_new_tokens=(data["max_tokens"])
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(data.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        output_json = {
            "object": "chat.completion",
            "model": self.model_name,
            "choices": [
                {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response,
                    "refusal": None,
                    "annotations": []
                },
                "logprobs": None,
                "finish_reason": "stop"
                }
            ]}
        return output_json
