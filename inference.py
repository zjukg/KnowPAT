import os
import json
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer



base_path = "YOUR MODEL PATH"


def load_test_dataset():
    test_file_path = "YOUR TEST DATA PATH"
    test_file = open(test_file_path, "r")
    dataset = []
    for line in test_file.readlines():
        record = json.loads(line)
        dataset.append(record)
    return dataset



if __name__ == "__main__":
    test_dataset = load_test_dataset()
    tokenizer = LlamaTokenizer.from_pretrained(base_path)
    cuda = 'cuda:0'
    checkpoint_name = "YOUR SAVE PATH"
    lora_weights = "".format(checkpoint_name)
    model = LlamaForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16
    ).cuda(cuda)
    model = PeftModel.from_pretrained(model, lora_weights)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model = model.eval()
    result = []
    for data in test_dataset:
        inputs = data["query"]
        answer = data["responses"][0]
        prompt = inputs
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.cuda(cuda)
        generate_input = {
            "input_ids":input_ids,
            "max_new_tokens":192,
            "do_sample":True,
            "top_k":50,
            "top_p":0.95,
            "temperature":0.3,
            "repetition_penalty":1.3,
            "eos_token_id":tokenizer.eos_token_id,
            "bos_token_id":tokenizer.bos_token_id,
            "pad_token_id":tokenizer.pad_token_id
        }
        generate_ids = model.generate(**generate_input)
        context = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = response.replace(context, "").strip().replace("Answer：", "")
        print(response + '\n\n\n')
        result.append(
            {
                "question": prompt,
                "answer": answer,
                "predict": response
            }
        )
    # print(result)
    json.dump(result, open("test_{}.json".format(checkpoint_name), "w"), ensure_ascii=False)
