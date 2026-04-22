from PIL.ImageChops import offset

from inferscale.best_of_n import BestOfNSampler
from datasets import load_dataset
if __name__ == '__main__':
    # models_names = ["Sachin21112004/distilbart-news-summarizer","google/pegasus-xsum"]
    model_name = "Sachin21112004/distilbart-news-summarizer"
    evaluation_metric = "rougeL"
    bon = BestOfNSampler(model_name=model_name,evalution_metric=evaluation_metric)
    dataset = load_dataset("cnn_dailymail", "3.0.0")
    offset = 10
    queries = [dataset["train"][offset]["article"],
               dataset["train"][offset+1]["article"],
               dataset["train"][offset+2]["article"]]
    results = bon.generate(queries=queries,n=3)
    print(results)
