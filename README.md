# InferScale is an open-source inference scaling platform for large language models.

It improves LLM output quality using inference-time scaling techniques such as Best-of-N sampling, output scoring, and model ensembling. Instead of relying on a single generation, InferScale produces multiple candidate outputs and selects the highest-quality result using configurable scoring and selection logic.

This approach improves correctness, reliability, and robustness without retraining or modifying the base model. InferScale is model-agnostic and works with proprietary APIs, open-source models, and local deployments.

InferScale treats inference as an optimization problem: generate candidate outputs, evaluate them, and select the best one. This enables systematic quality improvements while keeping infrastructure simple and cost-efficient.

InferScale is designed for production use in AI agents, reasoning systems, enterprise LLM applications, and any system where output quality is critical.

# Main Resources

1. https://open.substack.com/pub/sebastianraschka/p/categories-of-inference-time-scaling?utm_campaign=post-expanded-share&utm_medium=web
2. https://arxiv.org/abs/2510.10787
3. https://medium.com/@adnanmasood/inference-time-scaling-how-modern-ai-models-think-longer-to-perform-better-a1e1a8155fbd
