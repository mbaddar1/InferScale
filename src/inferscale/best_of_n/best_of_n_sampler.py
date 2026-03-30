import sys
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
    SUPPORTED_MODELS = ["Sachin21112004/distilbart-news-summarizer", "google/pegasus-xsum"]

    def __init__(self, model_name:str):
        """
            Best-of-N inference sampler.

            Generates multiple responses from different models
            and selects the best candidate using cosine similarity.

            Parameters
            ----------
            model_name : list[str]
                List of model names used for generation.
            """
        if model_name in BestOfNSampler.SUPPORTED_MODELS:
            if model_name not in self.SUPPORTED_MODELS:
                raise ValueError("Invalid model: {}".format(model_name))
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=True)
        # TODO : (Investigate) use_fast seems to have no effect, with or without it I can see .is_fast is True
        self.eval_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("BestOfNSampler successfully initialized")

    def generate(self, queries: List[str], n: int) -> List[dict]:
        # TODO (Later Improvements)
        #   1. Use lambda map to reduce loops
        #   2. Support parallel processing (parallel sampling)
        #   3. Smart model selector (based on query intent identification and the task at hand
        # Batch Encoding
        repeated_queries = result = [q for q in queries for _ in range(n)]
        batch_encodings = self.tokenizer(repeated_queries, return_tensors="pt", max_length=1024, padding=True,truncation=True)
        batch_encodings_list = list(batch_encodings["input_ids"])

        response_token_ids = list(map(lambda x:self.model.generate(x.view(-1,1).T,
                            max_length=150,
                            min_length=40,
                            no_repeat_ngram_size=3,
                            length_penalty=2.0,
                            num_beams=4,
                            do_sample=True,
                            top_p=0.95,
                            temperature=1.5,
                            early_stopping=True),batch_encodings_list))
        decoded_results = list(map(lambda x:self.tokenizer.decode(x.view(-1),skip_special_tokens=True),response_token_ids))
        print(len(decoded_results))
        sys.exit(-1)
        for u, q in tqdm(enumerate(queries), desc="queries"):
            candidates = []
            for i in tqdm(range(n), desc="response-samples"):
                inputs = self.tokenizer(q, return_tensors="pt", max_length=1024, truncation=True)
                try:
                    summary_ids = self.model.generate(inputs["input_ids"],
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
                    result = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    candidates.append(result)
                except Exception as e:
                    logger.error(f"Exception : For query = {q}, model = {self.model_name},"
                                 f"sample # {i},exception = {e}")
            max_score = -np.inf
            sorted_candidates = []
            if len(candidates) > 0:
                doc_emb = self.eval_embedding_model.encode([q] * n)
                sum_emb = self.eval_embedding_model.encode(candidates)
                eval_scores = cosine_similarity(sum_emb, doc_emb)[:, 0]
                sorted_candidates = [c for _, c in sorted(zip(eval_scores, candidates), reverse=True)]
            else:
                logger.warning(f"No candidate for query {q}")
            return sorted_candidates
        #
        #         #
        #         # max_eval_score_index = np.argmax(eval_score)
        #         # max_eval_score_per_model = eval_score[max_eval_score_index]
        #         # if max_eval_score_per_model > max_score:
        #         #     max_score = max_eval_score_per_model
        #         #     best_candidate = candidates[max_eval_score_index]
        #         #     best_model = self.models_names[j]
        #     else:
        #         logger.warning("No candidates generated for query = {}".format(q))
        # best_results.append({"response": best_candidate, "score": float(max_score), "model": best_model})
        # return best_results
