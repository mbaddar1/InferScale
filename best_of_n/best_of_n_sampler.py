from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import logging as transformers_logging
from loguru import logger
from bert_score import score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

transformers_logging.set_verbosity_info()
transformers_logging.enable_progress_bar()
"""
This class is inspired by the BestOfNSampler in TRL/HuggingFace
https://huggingface.co/docs/trl/main/en/best_of_n
"""
# TODO Dev
#   1. Perturbation param to generate different summarization each time
#       best_of_n/best_of_n_sampler.py:41
class BestOfNSampler:
    SUPPORTED_MODELS = ["Sachin21112004/distilbart-news-summarizer"]
    def __init__(self,models_names:List[str],evaluation_metric:str):
        for model_name in models_names:
            if model_name not in self.SUPPORTED_MODELS:
                raise ValueError("Invalid model: {}".format(model_name))
            self.models = []
            self.tokenizers = []
            for model_name in models_names:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.models.append(model)
                self.tokenizers.append(tokenizer)
            self.evaluation_metric = evaluation_metric
            self.eval_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("BestOfNSampler successfully initialized")

    def generate(self,queries:List[str],n:int) -> dict:
        # TODO (Later Improvements)
        #   1. Use lambda map to reduce loops
        #   2. Support parallel processing (parallel sampling)
        #   3. Smart model selector (based on query intent identification and the task at hand
        for u,q in enumerate(queries):
            for j in range(len(self.models)):
                candidates = []
                for i in range(n):
                    tokenizer = self.tokenizers[j]
                    model = self.models[j]
                    inputs = tokenizer(q, return_tensors="pt", max_length=1024, truncation=True)
                    summary_ids = model.generate(
                        inputs["input_ids"],
                        max_length=150,
                        min_length=40,
                        no_repeat_ngram_size=3,
                        length_penalty=2.0,
                        num_beams=4,
                        do_sample=True,
                        top_p=0.95,
                        temperature=1.5,
                        early_stopping=True
                    )

                    result = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    candidates.append(result)
                    # print(f"query # {u},sample # {i+1} , len(q) = {len(q)}, len(result) = {len(result)}")
                # print("Calculating scores") # FIXME - bertscore calculation stucked
                # bert_score_vals = score(cands=candidates,refs=refs,lang="en",model_type="bert-base-uncased")
                # print(bert_score_vals)

                doc_emb = self.eval_embedding_model.encode([q]*n)
                sum_emb = self.eval_embedding_model.encode(candidates)
                eval_score = cosine_similarity(sum_emb, doc_emb)[:,0]
                print(f"q # {u} , eval_score : {eval_score}")