# Inferscale : Lightweight package for inference-time LLM scaling

## Overview

InferScale is a lightweight Python package for smarter and more cost-efficient inference in AI applications.

It is designed to support two core capabilities:

1. **Adaptive model selection at inference time**  
   InferScale can choose the most appropriate model for a given task and input query, helping developers route requests more intelligently.

2. **Inference-time scaling for higher-quality outputs**  
   InferScale can aggregate multiple candidate outputs and select the best result, improving response quality without depending solely on larger or more expensive models.
## Getting Started
This section helps the developer to quickly getting started to develop a simple code with `inferscale`
## Installation

```bash
pip install inferscale datasets sentence-transformers rich
```

## Simple example 
The current scope of `inferscale` is focusing on summarization tasks. This code example loads to summarization models, generates 3 sample-responses from each model and selects the response with the highest score. Currently we support cosine similarity between the response and the input text as a scoring measure for summarization quality. 
```aiignore
import json
from inferscale.best_of_n import BestOfNSampler
from datasets import load_dataset
from rich import print_json
if __name__ == '__main__':
    models_names = ["Sachin21112004/distilbart-news-summarizer","google/pegasus-xsum"]
    bon = BestOfNSampler(models_names=models_names)
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    queries = [dataset["train"][0]["article"],dataset["train"][1]["article"],dataset["train"][2]["article"]]
    results = bon.generate(queries=queries,n=3)
    print_json(json.dumps(results,indent=4))
```
