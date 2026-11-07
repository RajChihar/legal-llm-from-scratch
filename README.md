# CS781 Capstone: In-House Legal Intelligence System

A compact legal language model trained **from scratch** on Indian legal text (Phase 1), then adapted to six downstream legal NLP tasks (Phases 2–7). No pretrained language models or external legal corpora were used for the foundation model; every backbone in Phases 2–7 is initialised from a Phase-1 checkpoint.

**Author:** Rajkumar Chihar (261110049), IIT Kanpur · **Codabench:** `r_261110049`

| Phase | Task | Primary metric | Codabench rank | Best local dev score |
|---|---|---|---|---|
| 1 | Legal LM pre-training (128-word continuation) | ROUGE-L | 51 | ROUGE-L 0.308, BLEU-4 19.8 |
| 2 | Legal Named Entity Recognition (12 types) | strict span macro-F1 | 67 | macro-F1 0.579 |
| 3 | Rhetorical role prediction (CL + IT) | macro-F1 | 49 | macro-F1 0.569 |
| 4 | Court judgment prediction | macro-F1 | 59 | macro-F1 0.711 |
| 5 | Legal statute identification (100 statutes) | macro-F1 | 51 | macro-F1 0.400 |
| 6 | Abstractive summarization | BERTScore-F1 | 57 | BERTScore-F1 0.844 |
| 7 | Prior case retrieval | micro-F1@7 | 42 | micro-F1@7 0.388 |

The full write-up is in [`Report/`](Report/).

---

## Repository layout

```
Phase 1/   tokenizer notebook + one notebook per architecture (A, A v2, B, C, D); Kaggle run copies in A/ B/ C/ D/ "Fine tunned"/
Phase2/    Legal NER            — CS781_Phase2_LNER_v2.ipynb, package p2lner/, configs/, tests/
Phase 3/   Rhetorical roles     — CS781_Phase3_RR_v2.ipynb,   package p3rr/,   tests/
Phase 4/   Judgment prediction  — CS781_Phase4_CJPE.ipynb,    package p4cjpe/, tests/
Phase 5/   Statute ID           — CS781_Phase5_LSI.ipynb,     package p5lsi/,  tests/
Phase 6/   Summarization        — CS781_Phase6_Summ_RM.ipynb, package p6summ/, tests/
Phase 7/   Prior case retrieval — CS781_Phase7_PCR.ipynb,     package p7pcr/,  tests/
Report/    final report
```

In every phase folder:

- `CS781_*.ipynb` is the source notebook. It writes its Python package to disk with `%%writefile` cells, so **the notebook alone is enough to reproduce the phase**; the package folder next to it holds the same code for reading and testing.
- `phase-*.ipynb` is the executed Kaggle run with all outputs (training logs, tables, plots).
- `results*/` holds the run summaries: `RESULTS.csv` (one row per experiment), per-run `summary.json`, `history.json`, dev reports and the submitted `prediction.json` files.

Model checkpoints (`*.pt`), datasets and caches are **not** stored in this repository because of their size (about 35 GB).

---

## Phase 1: Foundation model

### Tokenizer
Byte-level BPE trained from scratch on the official Phase-1 corpus (`Phase1_0_BPE_Tokenizer_and_Data.ipynb`): vocabulary **32,000**, Unicode-aware pre-tokenisation (`regex` with `\p{L}`, `\p{N}`), special tokens `<eot> <pad> <bos> <sep> <mask>`. The same `tokenizer.json` is shared by all architectures and all later phases.

### Architectures
All models use RMSNorm, SwiGLU feed-forward layers (multiplier 8/3), rotary position embeddings and no bias terms; the decoder models tie input and output embeddings. All parameter counts are inside the required 50M–300M range.

| Model | Design | Shape | Params |
|---|---|---|---|
| **A v2** (final) | LLaMA-style decoder, grouped-query attention | 11 layers, d=768, 12 query / 4 KV heads, context 1024 | 93.8M |
| A | LLaMA-style decoder, GQA | 12 layers, d=640, 10 / 2 heads, context 1024 | 72.1M |
| B | Encoder–decoder | 6 + 6 layers, d=576, encoder 768 / decoder 384 tokens | 74.2M |
| C | Hybrid prefix-LM + MLM | 10 layers, d=640, context 1024 | 70.1M |
| D | Efficient: sliding-window attention + multi-query attention | 12 layers, d=640, window 256, context 2048 | 71.1M |

### Training
Each model was trained on a **400M-token** budget with AdamW (β = 0.9 / 0.95, weight decay 0.1), peak learning rate 6e-4 with 500 warm-up steps and cosine decay to 6e-5, gradient clipping at 1.0, and a micro-batch of 8 with 8 accumulation steps (about 65k tokens per step, 6,103 steps). Training ran on two Kaggle T4 GPUs (`DataParallel`) with fp16 autocast and a gradient scaler, saving a checkpoint every 500 steps so that killed sessions can resume.

| Model | Final val. loss | ROUGE-L | BLEU-4 |
|---|---|---|---|
| A v2 | 1.955 | **0.308** | **19.8** |
| A | **1.936** | 0.287 | 17.7 |
| D | 1.984 | 0.274 | 16.3 |
| C | 2.132 | 0.274 | 16.2 |
| B | 2.207 | 0.177 | 3.4 |

ROUGE-L and BLEU-4 are measured on 300 held-out prompts.

**Decoding for submission:** temperature 0.6, top-k 50, top-p 0.95, repetition penalty 1.1 (window 256), up to 256 new tokens, trimmed to the 128-word target. Six decoding settings were compared on the held-out prompts.

---

## Shared fine-tuning toolkit (Phases 2–7)

Every task package contains the same building blocks:

- **Backbone loading** from any Phase-1 checkpoint, with optional bidirectional attention for the decoder models and layer truncation for small students.
- **Parameter-efficient fine-tuning implemented from scratch:** LoRA, rsLoRA, LoRMA, QLoRA (NF4), AdaLoRA, Houlsby and Pfeiffer adapters, prefix tuning, prompt tuning, IA³ and BitFit, alongside full and partial fine-tuning.
- **Knowledge distillation** from the best large model into a 4–6-layer student.
- **Parallel experiment runner:** each experiment is a config; experiments run as subprocesses spread over the two Kaggle GPUs, with a per-run time limit and resume support.
- **Checked submissions:** every `prediction.json` is validated (IDs, schema, value ranges) before it is zipped.

---

## Phase 2: Legal NER
Token classification with BIO tags over 512-token windows, followed by exact character-offset reconstruction from the tokenizer's byte offsets; on dev, converting gold spans to BIO and back gives macro-F1 0.994, so alignment loses almost nothing. Decoding uses constrained BIO Viterbi plus trimming rules for punctuation and whitespace.

**Best run:** `E3_A2_lora_bidir` — A v2 backbone with bidirectional attention and LoRA (r = 8) on attention and FFN, linear head; dev macro-F1 **0.579**. Also compared: full fine-tuning of C with a linear or CRF head, LoRMA on B, an adapter on D and a distilled 4-layer student.

## Phase 3: Rhetorical role prediction
Each sentence is encoded inside a 1024-token context window (up to 128 tokens per sentence), a GRU runs over the sentence vectors of the document, and a CRF predicts the role sequence. Corporate-law (CL) and IT documents are trained jointly.

**Best run:** `V1_A2_lora_crf_1k` — A v2 + LoRA + GRU + CRF; dev macro-F1 **0.569**. Removing document context (sentence-only encoding) drops macro-F1 to 0.357.

## Phase 4: Court judgment prediction
Long judgments are split into **4 × 512-token chunks** taken from the head and the tail of the document. Attention pooling inside each chunk and attention aggregation across chunks (with position buckets) feed a binary classifier, and the decision threshold is tuned on dev.

**Best run:** `R1_A2_full_hier` — A v2, full fine-tuning; dev macro-F1 **0.711** (threshold 0.51). A distilled 6-layer D student reaches 0.706 with half the parameters.

## Phase 5: Legal statute identification
The same hierarchical document encoder as Phase 4, with a 100-way multi-label head trained with binary cross-entropy and **per-label thresholds** tuned on dev. Label-wise attention, asymmetric loss and QLoRA were also compared.

**Best run:** `T1_A2_full_hier` — A v2, full fine-tuning; dev macro-F1 **0.400**, micro-F1 0.494.

## Phase 6: Abstractive summarization
Decoder-only summarization with the A v2 model: the source (1,400 tokens) and the summary (up to 640 tokens) share one sequence, with RoPE position interpolation (×2) to fit 2,048 positions. Source-selection strategies (head, tail, head+tail, uniform, salient sentences) were compared; decoding is greedy with no repeated 4-grams and a minimum length. The encoder–decoder B model was tested with fusion-in-decoder over 3,072 source tokens. Extra submissions: an MBR ensemble, and a reward model trained on preference pairs built from the official dev summaries, used for best-of-N reranking.

**Best run:** `X5_A2_lorma_salient` — A v2 + LoRMA (r = 16) + salient context selection; dev BERTScore-F1 **0.844**, ROUGE-1 / 2 / L 0.478 / 0.230 / 0.250.

## Phase 7: Prior case retrieval
Bi-encoder dense retrieval: queries and candidates are encoded by the same backbone as 4 × 512-token chunks (2 chunks sampled per document during training), pooled and projected to 256 dimensions, and scored with cosine similarity or chunk-level MaxSim. Training uses InfoNCE (τ = 0.05) with in-batch and BM25-mined hard negatives and false-negative masking. A from-scratch BM25 (unigrams + bigrams) is tuned on dev and fused with the dense scores; the top 7 candidates are submitted.

**Best submission:** fusion of three dense models (`X7_A2_maxsim`, `R3_C_partial`, `X8_A2_ance`) with BM25 (weight 0.3); dev micro-F1@7 **0.388**, against 0.310 for BM25 alone and 0.326 for the best single dense model.

---

## How to run (Kaggle)

1. **Phase 1:** run `Phase1_0_BPE_Tokenizer_and_Data.ipynb` once to build the tokenizer and the tokenized corpus, then any `Phase1_*` architecture notebook. Save the run folders (each contains `model_final.pt` and `tokenizer.json`) as a Kaggle dataset, for example `cs781-phase1-ckpts` with sub-folders `A_llama_decoder_v2/`, `B_encoder_decoder/`, `C_hybrid_prefix_mlm/` and `D_efficient_swa_mqa/`.
2. **Phases 2–7:** import the phase's `CS781_*.ipynb` into Kaggle and set
   - **Accelerator:** GPU T4 × 2,
   - **Internet:** on,
   - **Input:** the Phase-1 checkpoint dataset (checkpoints are found by folder name anywhere under `/kaggle/input`),
   - **Secret:** `HF_TOKEN`, a Hugging Face token with access to `Exploration-Lab/CS-781-Capstone` (the notebook downloads its phase's data automatically).
3. Edit only the **SETTINGS** cell:
   - `SMOKE_TEST = True` runs every experiment on a tiny subset in a few minutes; do this first.
   - `RUN_LIST` chooses which experiments to run, `MAX_MINUTES` caps each run, `SKIP_FINISHED` resumes after a killed session, and `RUN_TESTS` runs the unit tests first.
4. Set `SMOKE_TEST = False` and use **Save Version → Save & Run All**. Results appear in `/kaggle/working/RESULTS.csv`, and checked submission ZIPs in `/kaggle/working/submissions/`.

### Running the tests locally

```bash
cd "Phase 7"
python -m pytest -q tests
```

The unit tests build tiny random models, so they run on a CPU in seconds. The checkpoint test needs the real Phase-1 checkpoints and reports an error when they are not available.

---

## Requirements
Python 3.10+, PyTorch 2.x, NumPy, `regex` and `huggingface_hub`; Phase 6 also uses `rouge_score` and `bert_score` for local evaluation. Everything else (tokenizer, models, PEFT methods, BM25, metrics) is implemented in this repository.
