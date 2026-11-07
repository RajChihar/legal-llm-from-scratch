# CS781 Capstone · Phase 2 · Legal Named Entity Recognition

Fine-tunes the four Phase-1 legal language models (A decoder, B encoder-decoder, C hybrid, D efficient) for
span-level legal NER, scored with **strict macro-F1**: start, end and label must all match exactly.

## Layout
```
Phase2/
├── p2lner/                  the pipeline (importable package)
│   ├── tokenizer.py         Phase-1 byte-level BPE (load-only) + per-token character offsets
│   ├── data.py              HF download (gated), loading, gold-format detection, dev split, EDA
│   ├── spans.py             char spans <-> BIO tags, BIO-constrained Viterbi, learned boundary trimming,
│   │                        round-trip reachability check
│   ├── metrics.py           strict P/R/F1 (per label, macro, micro) + error analysis
│   ├── phase1_models/       Phase-1 model classes copied verbatim (so checkpoints load unchanged)
│   ├── backbones.py         checkpoint -> token encoder; causal or bidirectional attention, padding masks
│   ├── peft.py              full / partial / frozen / LoRA / LoRMA(+, π) / Houlsby adapter / BitFit / prompt tuning
│   ├── heads.py             linear head, linear-chain CRF
│   ├── model.py             NERModel + losses (CE, class weights, focal, knowledge distillation)
│   ├── train.py             sliding windows, training loop, inference, evaluation, submission zip
│   └── run.py               CLI
├── configs/                 one JSON per experiment (E1–E6 required set, X_* ablations)
├── tests/test_pipeline.py   correctness checks (run before trusting any number)
└── Phase2_LNER_Kaggle.ipynb driver notebook: data → EDA → train → error analysis → submission
```

## Pipeline
1. **Tokenise with offsets.** Each token gets its character span. The *content span* (whitespace stripped) defines
   entity boundaries: `" Court"` starts at `C`.
2. **Gold → BIO.** 12 labels → 25 tags. Whitespace-only tokens are transparent: they never start or end an entity.
   `fine_punct=True` splits punctuation runs (`".,"` → `"."`, `","`) so entities that end inside such runs become reachable.
   The EDA cell reports the round-trip ceiling for both settings.
3. **Windows.** Documents are longer than the context, so they are cut into windows of `max_len` tokens with stride
   `stride`. At inference each token's scores are averaged over all windows that contain it, weighted by how much
   context the token had (distance to the window edge for bidirectional attention, position in the window for causal).
4. **Backbone.** A Phase-1 checkpoint with its LM head removed. `attention="bidirectional"` lets every token see the whole
   window. This is the natural mode for B's encoder and for C's prefix. For A and D it changes the pretraining
   regime, which is why `E3b` (causal) exists as an ablation. D keeps its sliding window in both directions.
5. **Adaptation** (`method`), **head** (linear / CRF), **loss** (CE, class weights, KD).
6. **Decoding.** BIO-constrained Viterbi (an `I-X` must follow `B-X`/`I-X`), then spans are trimmed of characters that
   (almost) never start or end a gold entity in the training data.
7. **Evaluation** on a fixed held-out 15% of documents (same seed for every run). The outputs are a per-label table and
   error categories (type / boundary / both / missed / spurious) with examples.
8. **Submission.** The test IDs must match exactly; `prediction.json` goes at the zip root.

## Usage
```bash
python tests/test_pipeline.py --ckpt_dir <folder with the Phase-1 run folders>
python -m p2lner.run --eda --config configs/E1_C_full_linear.json --set data_dir=<.../Phase_2_LNER>
python -m p2lner.run --config configs/E1_C_full_linear.json --set data_dir=<.../Phase_2_LNER> --set ckpt_root=<ckpts>
python -m p2lner.run --predict runs/E1_C_full_linear --set data_dir=<...>
```
Checkpoints are referenced as `@A`, `@A2`, `@B`, `@C`, `@D`. They are found under `ckpt_root` or `/kaggle/input` by
their Phase-1 run-folder name.

Each run writes the following to `runs/<run_name>/`: `config.json`, `model_info.json`, `history.json`, `best.pt` (trainable
weights only, except for full fine-tuning), `dev_scores.json`, `dev_errors.json`, `dev_prediction.json`, `summary.json`
(params, trainable %, train time, peak memory, inference speed) and `submission/submission.zip`.

## Methods (references)
LoRA (Hu et al., 2021) · LoRMA (Bihany, Patel & Modi, 2025, arXiv:2506.07621) · Houlsby adapters (Houlsby et al., 2019) ·
BitFit (Ben Zaken et al., 2022) · Prompt tuning (Lester et al., 2021) · Knowledge distillation (Hinton et al., 2015) ·
CRF (Lafferty et al., 2001).
