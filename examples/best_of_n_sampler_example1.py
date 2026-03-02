from best_of_n import BestOfNSampler
if __name__ == '__main__':
    models_names = ["Sachin21112004/distilbart-news-summarizer"]
    evaluation_metric = "bert_score"
    bon = BestOfNSampler(models_names=models_names,evaluation_metric=evaluation_metric)
    queries = ["A new book lifts a lid on how, in May 2025, Pope Leo XIV was elected as the first US-born pope in the Catholic Church’s 2,000-year history. Its authors tell in previously unheard detail how Cardinal Robert Prevost, a low-key Augustinian friar from Chicago, had quietly garnered support from fellow cardinals as the conclave got underway but remained under the radar of wider attention as a serious candidate.",
               "Colin Gray, the teen’s father, has pleaded not guilty to charges of murder and manslaughter. Prosecutors say he acted recklessly by buying his son the rifle as a Christmas gift and allowing him access to it despite previous warnings that his son was a danger to others. His defense has said he was unaware his son was planning the shooting and had taken steps to try to get him help.",
               "But the campaign has revealed something about Cornyn too. After more than two decades in the Senate, cordially climbing the rungs, backing conservative policies but also working on bipartisan deals, Cornyn is putting aside that history. He’s presenting himself as a stalwart ally of President Donald Trump — despite periodically breaking with him over the years — and spending furiously to try to drag Paxton down."]

    bon.generate(queries=queries,n=3)
