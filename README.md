# GenMux

**GenMux** is an open-source orchestration layer for **selecting, routing, ranking, and mixing outputs from multiple generative models** — including **LLMs, diffusion models, and multimodal systems**.

Modern generative applications rarely rely on a single model. GenMux provides a control plane that allows you to **compose heterogeneous generators**, evaluate their outputs, and produce a **single, optimized response** using configurable strategies.

---

## Why GenMux?

Generative models have complementary strengths:
- Some LLMs reason better
- Some generate higher-quality text
- Diffusion models excel at visual fidelity
- Multimodal models bridge text, image, and audio

**GenMux lets you use them together — deliberately, transparently, and reproducibly.**

---

## Core Capabilities

- **Model Routing**  
  Route prompts to one or more generative models based on task, modality, or policy.

- **Output Selection & Ranking**  
  Score, filter, and rank candidate outputs using rule-based, heuristic, or learned evaluators.

- **Output Mixing & Fusion**  
  Combine multiple model outputs into a single response (e.g. voting, merging, refinement, chaining).

- **Evaluation-Aware Generation**  
  Treat evaluation as a first-class component in the generation loop.

- **Model-Agnostic Design**  
  Works across vendors, open-source models, and custom inference backends.

---

## Typical Use Cases

- LLM routing and arbitration across providers
- Hybrid pipelines combining **LLMs + diffusion models**
- Ensemble generation for higher reliability and quality
- Research on evaluation-driven generation and model selection
- Production systems requiring control, observability, and reproducibility

---

## Design Philosophy

- **Explicit over implicit** — decisions are visible and configurable  
- **Evaluation first** — generation without scoring is guesswork  
- **Composable primitives** — build simple flows or complex pipelines  
- **Production-minded** — deterministic, inspectable, and testable  

---

## Status

🚧 **Early stage / actively evolving**

APIs and internals may change as the project stabilizes.

---

## Roadmap (High-Level)

- Unified model interface for LLMs and diffusion models  
- Pluggable evaluation modules (rules, metrics, LLM-based judges)  
- Routing policies and selection strategies  
- CLI and Python SDK  
- Experiment tracking and observability hooks  

---

## Inspiration & Prior Work

GenMux draws inspiration from ensemble learning, mixture-of-experts systems, model routing, and evaluation-driven generation research.

GenMux is **not tied to a single paper or model** and aims to serve as a general-purpose orchestration layer for generative systems.

---

## License

[Add your license here]
