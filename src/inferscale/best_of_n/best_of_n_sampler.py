from typing import List

import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import logging as transformers_logging
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
transformers_logging.set_verbosity_info()
transformers_logging.enable_progress_bar()
"""
This class is inspired by the BestOfNSampler in TRL/HuggingFace
https://huggingface.co/docs/trl/main/en/best_of_n
"""
# TODO Dev (each line with a smaller version)
#   1. Perturbation param to generate different summarization each time DONE
#       best_of_n/best_of_n_sampler.py:41
#   2. Debug why all best results from one model
#   2. Use Emb and cos similarity DONE
#   . Bug - Why Google pegasus model is unstable
#   . Add verbosity levels
#   . Add eval metrics
#   . make dep install auto
#   2. Add resilience to model pretrained load, verbose info and TIMEOUT mechanism TODO
#   3. Test with larger text DONE
#   . Bug - Problem with some long queries, vocab out of index
#   4. Tune generate params
#   5. Document dev steps in the ticket - You build things
#   6. Add time performance meta data (normalize per token)
#   7. Add GPU support
#   8. Better logging
#   9. Add requirements.txt
class BestOfNSampler:
    SUPPORTED_MODELS = ["Sachin21112004/distilbart-news-summarizer","google/pegasus-xsum"]
    def __init__(self,models_names:List[str]):
        for model_name in models_names:
            if model_name not in self.SUPPORTED_MODELS:
                raise ValueError("Invalid model: {}".format(model_name))
        self.models_names = models_names
        self.models = []
        self.tokenizers = []
        for model_name in models_names:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.models.append(model)
            self.tokenizers.append(tokenizer)
        self.eval_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("BestOfNSampler successfully initialized")

    def generate(self,queries:List[str],n:int) -> List[dict]:
        # TODO (Later Improvements)
        #   1. Use lambda map to reduce loops
        #   2. Support parallel processing (parallel sampling)
        #   3. Smart model selector (based on query intent identification and the task at hand
        best_results = []
        for u,q in tqdm(enumerate(queries),desc="queries"):
            max_score = -np.inf
            best_candidate = None
            best_model = None
            for j in tqdm(range(len(self.models)),desc="models"):
                candidates = []
                for i in tqdm(range(n),desc="response-samples"):
                    tokenizer = self.tokenizers[j]
                    model = self.models[j]
                    inputs = tokenizer(q, return_tensors="pt", max_length=1024, truncation=True)
                    try:
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
                    except Exception as e:
                        logger.error(f"Exception : For query = {q}, model = {self.models_names[j]}, sample # {i},exception = {e}")

                if len(candidates)>0:
                    doc_emb = self.eval_embedding_model.encode([q]*n)
                    sum_emb = self.eval_embedding_model.encode(candidates)
                    eval_score = cosine_similarity(sum_emb, doc_emb)[:,0]
                    max_eval_score_index = np.argmax(eval_score)
                    max_eval_score_per_model = eval_score[max_eval_score_index]
                    if max_eval_score_per_model > max_score:
                        max_score = max_eval_score_per_model
                        best_candidate = candidates[max_eval_score_index]
                        best_model = self.models_names[j]
                else:
                    logger.warning("No candidates generated for query = {}".format(q))
            best_results.append({"response":best_candidate,"score":float(max_score),"model":best_model})
        return best_results