import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Notice Preprocess class Must Be Named "Model"
class Model(object):
    
    def load_model(self, model_name: str, model_path: str, device: str, dtype: torch.dtype):
        """ Load and initialize your model

        Args:
            model_path (str): Local path of the model. Can be path to a folder or a file
            device (str): string of the model's device. It is determined by the inference system: auto, cuda:0, cuda:1, ..., cpu
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

    def preprocess(self, body: any, additional_params: dict = None) -> any:
        """ Preprocessing function

        Args:
            body (any): Any type of data may be accepted. Json and bytes data are tested and guaranteed to be processed from the API
            additional_params (dict): A dictionary of additional parameters (e.g., max_tokens, temperature, etc.)

        Returns:
            any: Preprocessed data, ready to be fed into the model
        """
        if 'messages' in body:
            messages = body['messages']
            model = body['model']
        else:
            prompts = body['prompts']
            model = body['model_name']
            messages = []
            for prompt in prompts:
                messages.append({
                    'role': prompt['role'],
                    'content': prompt['text']
                })

        if model != self.model_name:
            raise ValueError(f"Input model {model} is not compatible with served model {self.model_name}")
        
        # Process the messages with tokenizer
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Include additional params in the return (if any)
        if additional_params:
            model_inputs['additional_params'] = additional_params

        return model_inputs

    def process(self, data: any, additional_params: dict = None) -> dict:
        """ Main processing function

        Args:
            data (any): The output of the preprocess function.
            additional_params (dict): A dictionary of additional parameters like max_tokens, temperature, etc.

        Returns:
            dict: A dict type, json serializable output should be returned.
        """
        # Extract additional parameters
        max_tokens = additional_params.get("max_tokens", 4096)
        temperature = additional_params.get("temperature", 1.0)
        top_p = additional_params.get("top_p", 1.0)
        n = additional_params.get("n", 1)

        # Call model's generate method with the passed additional parameters
        generated_ids = self.model.generate(
            **data,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n
        )

        # Remove the input part from the output
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(data.input_ids, generated_ids)
        ]

        # Decode the output
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Prepare the response in the required format
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
            ]
        }
        
        return output_json
