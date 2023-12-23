import json
import torch

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

base_model = "YOUR MODEL PATH"

if __name__ == "__main__":
    dataset = open("dataset.json")
    i = 0
    triple_cls = torch.zeros((10000, 768))
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModel.from_pretrained(base_model)
    model = model.eval()
    model = model.cuda()
    with torch.no_grad():
        for line in dataset.readlines():
            record = json.loads(line)
            question = record['query']
            inputs = tokenizer(
                question,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to('cuda')
            triple_emb = model(**inputs)[0][:, 0]
            triple_cls[i] = F.normalize(triple_emb, dim=1)
            print(i)
            i += 1
    torch.save(triple_cls, open("question_embeddings_bge.pth", "wb"))

    