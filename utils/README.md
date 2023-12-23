# Triple Linking

## Pipeline
1. Before conduct the unsupervised triple linking, you need to prepare a bert-like pre-trained model ([BERT](https://github.com/huggingface/transformers), [SimCSE](https://github.com/princeton-nlp/SimCSE), [BGE](https://github.com/FlagOpen/FlagEmbedding)) as the textual encoder and change the  `base_model` variable in the scripts.
2. Run the scripts `triple_bge.py` and `question_bge.py` to encode the triples and questions using semantic embeddings and run the script `retrieve_relative_triple.py` to retrive the top-K relative knowledge for each question.


The format of the question data in `question_bge.py` is the same as the data sample in `data/` and we provide a KG data sample for reference.
