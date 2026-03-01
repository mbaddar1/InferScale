from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import logging as transformers_logging
from loguru import logger
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
    def __init__(self,models_names:List[str],evaluation_metrics:List[str]):
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
            self.evaluation_metrics = evaluation_metrics
            logger.info("BestOfNSampler successfully initialized")

    def generate(self,queries:List[str],n:int) -> dict:
        # TODO (Later Improvements)
        #   1. Use lambda map to reduce loops
        #   2. Support parallel processing (parallel sampling)
        #   3. Smart model selector (based on query intent identification and the task at hand
        for q in queries:
            for j in range(len(self.models)):
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
                        early_stopping=True
                    )
                    result = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    print(f"query = {q},sample # {i+1} , result = {result}")
