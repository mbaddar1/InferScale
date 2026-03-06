from inferscale.best_of_n import BestOfNSampler
from datasets import load_dataset
if __name__ == '__main__':
    models_names = ["Sachin21112004/distilbart-news-summarizer","google/pegasus-xsum"]
    bon = BestOfNSampler(models_names=models_names)
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    queries = [dataset["train"][0]["article"],dataset["train"][1]["article"],dataset["train"][2]["article"]]
    results = bon.generate(queries=queries,n=3)
    print(results)
