# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [0.1.4] - 01-06-2026
### To Add
* Benchmark 3 summarization models over news dataset using cosine similarity and SummaC. 
## [0.1.3] - 01-05-2026 (TODO)
### To Add 
* Use SummaC as a reference-free summarization quality metric. For more info see this [link](https://medium.com/@ankita.bagaria8/evaluation-methods-for-text-summarization-with-and-without-reference-summaries-66cb38505749)

## [0.1.2] - 01-04-2026
### Added 
* None
### Changed
* In previous version the tokenization was done iteratively with nested loops (iterate over input queries). 
  In this version both tokenization and sample generation (inference) is done via batching. 
* padding and truncation are used (max_len=1024), plan to make it parametric in upcoming versions.
See this [tutorial](https://www.kaggle.com/code/fiveflowerstarfish/transformers-tutorial-autotokenizer)
### Fixed 
* 
## [0.1.1] - 01-03-2026

### Added
- Simple approach for Best-Of-N Sampling. 
  - The scope of this version is for generating summary responses to a set of given text or queries $q_i, i=1,\dots,m$
  - Generate $n$ responses for each input query then for each query $q_i$ , the set of responses is ${resp_{i,1},\dots,resp_{i,j},\dots,resp_{i,n}}$
  - (Decreasingly) Sort the responses based on evaluation score $s(q_i,resp_{ij})$
  - For this version, cosine similarity between the embedding vector for $q_i$ and each response $response_{ij}$
    is used as a proxy for human evaluation. 
  - See this [paper](https://arxiv.org/pdf/2505.03481), page 5 , Quote "_We also used cosine similarity between sentence 
  embeddings of generated summaries and targets as this would measure the semantic similarity between source and target 
  irrespective of the words chosen._"
  - Currently, the code support one summarization model [(DistilBART-Based) News summarizer](https://huggingface.co/Sachin21112004/distilbart-news-summarizer)
  

### Fixed 
* None 
### Changed
* None
### Removed
* None
