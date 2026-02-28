# Pipeline Explanation (0 -> Chat Inference)

## Purpose
This file explains the full nanochat pipeline in a didactic way, from raw text to a working chat model.

Goal:
- make every stage explicit,
- show where compute/memory/time are spent,
- prepare clean links to future `PIPELINE_MATH/*` files.

---

## Step 0: Define The Target
Before code, define:
- target quality (for example, GPT-2 class behavior),
- hardware limit (single RTX 4090, 24GB VRAM),
- wall-clock budget (for example, <= 24h for pretrain + SFT + eval),
- acceptance metrics (loss, bpb, chat quality, stability).

Why this matters:
- every later decision (model size, tokens, seq length, optimizer, runtime tricks) depends on this budget.

---

## Step 1: Data Collection And Filtering
We gather training text and clean it.

Typical operations:
- collect corpora,
- remove duplicates,
- filter garbage/low-quality text,
- normalize formatting,
- optionally rebalance domains (code/web/books/chat/etc.).

Inputs:
- raw documents.

Outputs:
- cleaned text corpus.

Main cost profile:
- mostly CPU + disk I/O,
- memory pressure from dedup/filter indexing,
- little GPU usage.

Why it matters:
- bad data wastes expensive GPU steps,
- better data can improve quality at same compute.

---

## Step 2: Tokenizer Strategy
Text is converted into token IDs.

Options:
- reuse existing tokenizer,
- train a new tokenizer on your corpus.

Inputs:
- cleaned text.

Outputs:
- tokenizer model + vocab (and tokenized data later).

Main cost profile:
- tokenizer training: CPU-heavy preprocessing,
- tokenization pass: CPU + I/O throughput task.

Why it matters:
- tokenization quality changes sequence lengths and compression,
- this directly changes training cost and model behavior.

---

## Step 3: Dataset Building / Packing
Token streams are packed into training examples.

Typical operations:
- concatenate tokenized documents,
- chunk into sequences (`seq_len`),
- define train/val split,
- write binary shards for fast loading.

Inputs:
- tokenized corpus.

Outputs:
- packed train/val shards.

Main cost profile:
- CPU + disk bandwidth,
- storage layout quality affects training throughput.

Why it matters:
- bad packing increases padding/waste and dataloader stalls.

---

## Step 4: Model Configuration
We set architecture and runtime knobs.

Main knobs:
- `d_model`, `n_layers`, heads, FFN multiplier,
- context length and window pattern,
- attention implementation path (SDPA/FA2 when available),
- precision (`bf16`, etc.),
- optimizer/scheduler params.

Inputs:
- budget + desired quality.

Outputs:
- frozen config for experiment.

Main cost profile:
- no heavy compute yet, but this step controls all future FLOPs/VRAM.

Why it matters:
- wrong config can make training impossible or inefficient.

---

## Step 5: Pretraining (Largest Compute Stage)
Model learns next-token prediction on broad corpus.

Training loop (per step):
1. load batch of token sequences,
2. forward pass through transformer,
3. compute loss (cross-entropy),
4. backward pass (gradients),
5. optimizer update,
6. log metrics/checkpoints.

Inputs:
- model config + pretraining data.

Outputs:
- pretrained checkpoint(s).

Main cost profile:
- dominant GPU compute stage,
- dominant VRAM stage (activations, optimizer states),
- highly sensitive to kernel efficiency and memory bandwidth.

Why it matters:
- this stage creates most base capability.

---

## Step 6: Supervised Finetuning (SFT)
Model is tuned for instruction/chat behavior.

Typical operations:
- train on prompt-response style data,
- possibly mask loss to assistant tokens,
- run shorter targeted training than pretraining.

Inputs:
- pretrained checkpoint + instruction dataset.

Outputs:
- chat-aligned checkpoint.

Main cost profile:
- GPU compute, usually smaller than pretrain,
- still sensitive to seq length and batch strategy.

Why it matters:
- converts raw language ability into useful assistant behavior.

---

## Step 7: Evaluation
Measure quality and safety before inference.

Typical checks:
- validation loss / bpb / perplexity proxies,
- benchmark tasks (if configured),
- behavior sanity checks,
- regression checks against baseline.

Inputs:
- checkpoint(s), eval datasets.

Outputs:
- score reports and pass/fail decision.

Main cost profile:
- moderate GPU/CPU depending on eval size,
- often bottlenecked by eval protocol design.

Why it matters:
- avoids promoting faster but worse models.

---

## Step 8: Checkpoint Selection And Export
Pick the best checkpoint and prepare for serving.

Typical operations:
- select by metric + stability,
- save model/tokenizer/config artifacts,
- optionally convert format for deployment.

Inputs:
- training/eval artifacts.

Outputs:
- final inference-ready checkpoint package.

Main cost profile:
- mostly I/O/storage.

---

## Step 9: Inference Engine
Run autoregressive generation.

Per user request:
1. tokenize prompt,
2. run prefill forward pass,
3. generate token-by-token using KV cache,
4. decode tokens to text.

Inputs:
- final checkpoint + tokenizer + decoding params.

Outputs:
- generated text response.

Main cost profile:
- prefill: compute-heavy,
- decode: latency-sensitive, memory/cache-sensitive.

Why it matters:
- user-perceived speed depends mostly on this stage.

---

## Step 10: Chat UI Layer
User-facing interface around inference.

Typical operations:
- conversation state/history,
- system/user/assistant formatting,
- streaming tokens to frontend,
- basic controls (temperature, max tokens, reset chat).

Inputs:
- user message + model endpoint.

Outputs:
- interactive chat experience.

Main cost profile:
- mostly application/runtime/network overhead,
- model latency dominates total response time.

---

## End-To-End View
The full pipeline is:

1. target definition  
2. data collection/filtering  
3. tokenizer strategy  
4. dataset packing  
5. model configuration  
6. pretraining  
7. SFT  
8. evaluation  
9. checkpoint export  
10. inference engine  
11. chat UI

Most expensive stage:
- pretraining (by far).

Most common hard constraints:
- VRAM (training/inference),
- throughput (tokens/sec),
- data quality,
- optimization stability.

---

## Mapping To Future `PIPELINE_MATH`
Planned one-file-per-step mapping:

- `PIPELINE_MATH/00_target_budget.md`
- `PIPELINE_MATH/01_data_filtering_cost.md`
- `PIPELINE_MATH/02_tokenizer_cost.md`
- `PIPELINE_MATH/03_packing_io_cost.md`
- `PIPELINE_MATH/04_model_flops_vram.md`
- `PIPELINE_MATH/05_pretrain_step_math.md`
- `PIPELINE_MATH/06_sft_step_math.md`
- `PIPELINE_MATH/07_eval_math.md`
- `PIPELINE_MATH/08_export_cost.md`
- `PIPELINE_MATH/09_inference_prefill_decode_math.md`
- `PIPELINE_MATH/10_chat_serving_latency_budget.md`

Each file should include:
- formulas,
- variable definitions,
- example numbers from our real runs,
- cost type tags: `compute`, `memory`, `bandwidth`, `io`, `latency`,
- optimization targets and expected impact.
