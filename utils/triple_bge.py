import json
import torch

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

base_model = "YOUR MODEL PATH"

if __name__ == "__main__":
    triples = json.load(open("kgdata.json"))['triple']
    n = len(triples)
    triple_cls = torch.zeros((n, 768))
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModel.from_pretrained(base_model)
    model = model.eval()
    model = model.cuda()
    with torch.no_grad():
        for i in range(n):
            h, r, t = triples[i]
            input_sequence = "{} [SEP] {} [SEP] {}".format(h, r, t)
            inputs = tokenizer(
                input_sequence,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to('cuda')
            triple_emb = model(**inputs)[0][:, 0]
            triple_cls[i] = F.normalize(triple_emb, dim=1)
            print(i)
    torch.save(triple_cls, open("triple_embeddings-bge.pth", "wb"))

    