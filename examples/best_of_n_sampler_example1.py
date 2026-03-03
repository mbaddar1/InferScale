from best_of_n import BestOfNSampler
from datasets import load_dataset
if __name__ == '__main__':
    models_names = ["Sachin21112004/distilbart-news-summarizer","google/pegasus-xsum"]
    evaluation_metric = "bert_score"
    bon = BestOfNSampler(models_names=models_names,evaluation_metric=evaluation_metric)
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    # FIXME , this line leads to this error
    """
        dev_log/dev_log.txt
    """
    queries = [dataset["train"][0]["article"],dataset["train"][1]["article"],dataset["train"][2]["article"]]
    results = bon.generate(queries=queries,n=3)
    print(results)
