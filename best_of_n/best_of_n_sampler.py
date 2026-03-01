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
        for q in queries:
            pass