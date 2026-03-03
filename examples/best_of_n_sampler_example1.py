from best_of_n import BestOfNSampler
from datasets import load_dataset
if __name__ == '__main__':
    models_names = ["Sachin21112004/distilbart-news-summarizer","google/pegasus-xsum"]
    evaluation_metric = "bert_score"
    bon = BestOfNSampler(models_names=models_names,evaluation_metric=evaluation_metric)
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    max_len = 1000
    queries = [dataset["train"][0]["article"][:max_len],dataset["train"][1]["article"][:max_len],dataset["train"][2]["article"][:max_len]]
    results = bon.generate(queries=queries,n=3)
    print(results)
