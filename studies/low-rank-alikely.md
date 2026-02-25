Short answer: both exist, but dense is still the default core compute in most
  LLMs.

  - Most base models still do standard dense Transformer linear algebra in forward/
    backward (e.g., Mistral-style dense backbone with GQA/SWA optimizations, not
    low-rank factorized weights by default).
  - Low-rank is very common in fine-tuning, mainly via LoRA/QLoRA adapters, where
    the pretrained dense weights are frozen and low-rank updates are trained.
  - Some newer architectures use low-rank-like ideas internally (notably DeepSeek’s
    MLA compressing KV into latent representations).
  - MoE is another major trend, but that is sparse routing (active subset of
    experts), not the same thing as low-rank factorization of every dense weight
    matrix.

  So your mental model is right: for core training compute, dense matmuls are still
  the baseline; low-rank is mostly adaptation/compression, with a few architecture-
  level exceptions.

  Sources:

  - https://arxiv.org/abs/2106.09685 (LoRA)
  - https://arxiv.org/abs/2305.14314 (QLoRA)
  - https://arxiv.org/abs/2310.06825 (Mistral 7B)
  - https://arxiv.org/abs/2401.04088 (Mixtral MoE)
  - https://arxiv.org/abs/2412.19437 (DeepSeek-V3, MLA + MoE)
