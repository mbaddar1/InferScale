<p align="center">
  <img src="assets/inferscale_logo.png" width="300">
</p>

# What is InferScale

**InferScale** is a Python package that provides a unified and practical framework for applying *inference-time scaling* to large language models (LLMs).

At its core, InferScale moves beyond single-shot generation. It produces multiple candidate responses—either complete or partial—from one or more LLMs, and then intelligently **selects or aggregates** them to yield a higher-quality final output.

This strategy allows developers to enhance performance across a variety of tasks—including **text summarization, question answering, information extraction, and paraphrasing**—without the need for model fine-tuning or retraining.

By optimizing at inference time, InferScale offers a **scalable and cost-efficient alternative** to traditional training-heavy approaches. We view this paradigm as a highly practical and impactful way to improve LLM applications in real-world settings. This approach leverages **model diversity and response sampling** to increase the probability of obtaining a higher-quality output.

# Architecture

The current architecture of InferScale is shown below:

<img width="1200" height="327" alt="InferScale architecture" src="https://github.com/user-attachments/assets/1006af4b-4718-49a3-880c-389c3987be3d" />

Pipeline overview:

1. Multiple LLM models generate candidate responses
2. Each model can generate **N samples**
3. All responses are collected
4. A scoring mechanism selects the **best candidate**


# Proces overview

1. **Model Loading**  : The library loads __one__ model from a set predefind models for the given task (Text Summarization, Question Answering, Information Extraction, etc...). Currently we support generating different samples of responses from __one__ model , we plan at some point to add a model selector layer, generate and blend reponses from different models. 

2. **Generate Multiple Responses** : Each model generates __N candidate responses__ for the same input. The library is designed to get a batch of queries / input pieces of text. 

3. **Compute Semantic Similarity** :All responses are evaluated using an evaluation metric _that tries to mimic the human evaluation as much as possible_

4. **Generate the final response** :The response with the **highest score** is selected as the final output. (In the future we will implement appraoches the __blend__ top K responses from one or differnet models **stay tuned!!**)

# Details 
Currently we support the following **Tasks** to be done using __InferScale__ 
* Text Summarization


We plan to support the following tasks in the following releases
* Question-Answering
* Information Extraction
* Paraphrasing

### <span style="color:#fadb14;">▶</span>  Text Summarization + Inference-Scaling

The models we support currently (from the pool of models in Hugging Face) are `facebook/bart-large-cnn`, `sshleifer/distilbart-cnn-12-6`. Why ? If you see the [list](https://huggingface.co/models?pipeline_tag=summarization&sort=likes) of Hugging-Face summarization models, you will find that these are the most liked ones. We know this might be a naive approach for selecting the models to support, in the future we plan a more rigorous benchmarking. 

As __InferScale__ is designed to be "scalable", we focus , in the beginning, on the  __reference-free__ metrics for Automateed-Summarization Evaluation . You can check this simple [article](https://medium.com/@ankita.bagaria8/evaluation-methods-for-text-summarization-with-and-without-reference-summaries-66cb38505749) for more information. We support two metrics : 

* **Cosine Similarity** : We simple embed the query and result summarization using `all-MiniLM-L6-v2` then calculate `Cos(embedding(query),embedding(response))`

* **ROUGE** : This is one of the most classical metrics in text-summarization tasks . We apply ROUGE in and unsupervised (reference-free) fashion : we calculate ROUGE based on the input query as a reference text against the response as the predicted text. We use the `rouge-score` library in our current implementation. In the future, we will create our own to control the calculation poroces.

# Installation

`pip install inferscale datasets sentence-transformers rich`

# Example
```
import json
from inferscale.best_of_n import BestOfNSampler
from datasets import load_dataset
from rich import print_json


if __name__ == "__main__":

    # Candidate models
    model_names = [
        "Sachin21112004/distilbart-news-summarizer",
        "google/pegasus-xsum"
    ]

    # Initialize Best-of-N sampler
    bon = BestOfNSampler(models_names=model_names)

    # Load dataset
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    # Example queries
    queries = [
        dataset["train"][0]["article"],
        dataset["train"][1]["article"],
        dataset["train"][2]["article"]
    ]

    # Generate responses
    results = bon.generate(queries=queries, n=3)

    # Pretty print results
    print_json(json.dumps(results, indent=4))
```
# Change Log 
If you are intrested in the details of development and changes in each version, check the [CHANGE LOG](https://github.com/mbaddar1/InferScale/blob/main/changelog.md)
# Main Resources

1. https://open.substack.com/pub/sebastianraschka/p/categories-of-inference-time-scaling
2. https://arxiv.org/abs/2510.10787
3. https://medium.com/@adnanmasood/inference-time-scaling-how-modern-ai-models-think-longer-to-perform-better-a1e1a8155fbd
