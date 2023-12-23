import json
import torch
import numpy as np

from torch.nn.functional import cosine_similarity

def clean_kg(triples):
    s = set()
    cleanned_triple = []
    for triple in triples:
        text = "{} {} {}".format(triple[0], triple[1], triple[2])
        if text in s:
            continue
        else:
            s.add(text)
            cleanned_triple.append(triple)
    return cleanned_triple


if __name__ == "__main__":
    questions = open("dataset.json").readlines()
    triples = json.load(open("kgdata.json"))['triple']
    cleanned_triple = clean_kg(triples)
    question_embeddings = torch.load("question_embeddings_bge.pth").cuda(0)
    triple_embeddings = torch.load("triple_embeddings_bge.pth").cuda(0)
    assert len(questions) == question_embeddings.shape[0]
    assert len(triples) == triple_embeddings.shape[0]
    n = len(question_embeddings)
    m = len(triple_embeddings)
    k = 10
    result = []
    scores = question_embeddings @ triple_embeddings.T
    s = set()
    for i in range(n):
        score = scores[i]
        idxes = torch.argsort(score, descending=True)[: k]
        idxes = idxes.cpu().numpy().tolist()
        print(idxes)
        question_data = json.loads(questions[i])
        question_data["kg"] = []
        for j in range(k):
            question_data["kg"].append(triples[idxes[j]])
            s.add(idxes[j])
        result.append(question_data)
    json.dump(result, open("data-top{}.json".format(k), "w"), ensure_ascii=False)
    print(len(s), len(triples))
