from best_of_n import BestOfNSampler
if __name__ == '__main__':
    models_names = ["Sachin21112004/distilbart-news-summarizer"]
    evaluation_metrics = ["ROUGE"]
    bon = BestOfNSampler(models_names=models_names,evaluation_metrics=evaluation_metrics)
    queries = ["",""]
    bon.generate(queries=queries,n=10)
