# English–Hindi Neural Machine Translation from Scratch
### A from-scratch Transformer implementation and model-scaling study

![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-ee4c2c?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-English%E2%86%92Hindi%20Translation-1f6feb)
![Status](https://img.shields.io/badge/Status-Learning%20Project-orange)

> **Project positioning:** An educational engineering project that implements, trains, and evaluates encoder–decoder Transformers from randomly initialized weights. The final models are not production-quality translators; the purpose is to document the complete pipeline, experiments, engineering decisions, and lessons learned.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Goals and scope](#2-goals-and-scope)
3. [Dataset](#3-dataset)
4. [Token-length analysis and max length](#4-token-length-analysis-and-max-length)
5. [Tokenizer](#5-tokenizer)
6. [Transformer architecture](#6-transformer-architecture)
7. [Final model configurations](#7-final-model-configurations)
8. [Data pipeline and batching](#8-data-pipeline-and-batching)
9. [Training objective and optimization](#9-training-objective-and-optimization)
10. [Implementation from scratch](#10-implementation-from-scratch)
11. [GPU constraints and FlashAttention](#11-gpu-constraints-and-flashattention)
12. [Training results](#12-training-results)
13. [Interactive loss curves](#13-interactive-loss-curves)
14. [Decoding with beam search](#14-decoding-with-beam-search)
15. [Evaluation with BLEU](#15-evaluation-with-bleu)
16. [Qualitative translation examples](#16-qualitative-translation-examples)
17. [Interpretation and limitations](#17-interpretation-and-limitations)
18. [Engineering challenges and lessons learned](#18-engineering-challenges-and-lessons-learned)
19. [Repository structure](#19-repository-structure)
20. [Reproducibility checklist](#20-reproducibility-checklist)
21. [Future improvements](#21-future-improvements)
22. [Conclusion](#22-conclusion)

---

## 1. Project overview

This project builds an **English-to-Hindi neural machine translation system** using a custom encoder–decoder Transformer implemented in PyTorch. Instead of starting from a pretrained translation model, the final experiments trained the model weights from random initialization on parallel English–Hindi text.

The project covers the full path from paired-text processing to model training and generation:

- Shared Byte-Level BPE tokenization.
- Dataset filtering, dynamic padding, and attention masks.
- Transformer encoder and decoder modules.
- Multi-head self-attention and encoder–decoder cross-attention.
- Teacher-forced next-token training.
- Mixed-precision training, gradient accumulation, gradient clipping, and checkpointing.
- Autoregressive generation using beam search.
- Corpus BLEU evaluation with SacreBLEU.
- Comparison of two final model sizes and inspection of their loss curves and outputs.

**Important:** this is a learning and experimental project. Its BLEU scores and generated translations show that the final models did not reach the quality expected from a practical translation service.

## 2. Goals and scope

The main goals were to:

1. Understand and implement the encoder–decoder Transformer computation rather than only calling a high-level pretrained model.
2. Create a reproducible bilingual tokenizer and data pipeline.
3. Train a model to predict the next Hindi token conditioned on English input.
4. Handle practical constraints such as GPU memory, long sequences, and multi-hour training.
5. Evaluate models with both quantitative metrics and actual generated examples.
6. Compare model configurations and document the limitations honestly.

The final comparison focuses on the 23.36M- and 68.73M-parameter models. Earlier runs affected by a tokenizer issue are development history, not valid final model comparisons.

## 3. Dataset

The project uses the **IIT Bombay English–Hindi parallel corpus**. Each example contains an English sentence paired with a Hindi translation. This is supervised machine translation: the English sentence is the source sequence and the Hindi sentence is the target sequence.

### Raw split sizes reported by the token-length analysis

| Split | Number of pairs |
|---|---:|
| Train | 1,659,083 |
| Validation | 520 |
| Test | 2,507 |
| **Total raw pairs** | **1,662,110** |

These are the counts in the uploaded token-length report, before the final sequence-length filter. The final evaluation loader retained 2,394 test pairs, so the reported test BLEU scores are **not** scores over all 2,507 raw test pairs.

### Why parallel data matters

The model learns by observing source–target pairs. It is not simply learning Hindi language modeling: it must learn to use the English source to produce a Hindi sequence with corresponding meaning. Data cleanliness, alignment, tokenization, sentence length, and the diversity of the corpus all affect the result.

## 4. Token-length analysis and max length

The uploaded token-length statistics are reproduced below. Lengths are in the tokenizer’s token units, not whitespace-separated words.

| Split | Language | Sentences | Mean | Median | P95 | P99 | Maximum | Minimum |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Train | English | 1,659,083 | 19.74 | 14 | 54.00 | 92.00 | 3,228 | 2 |
| Train | Hindi | 1,659,083 | 49.65 | 33 | 147.00 | 246.00 | 5,337 | 2 |
| Validation | English | 520 | 26.44 | 24 | 50.00 | 67.81 | 105 | 8 |
| Validation | Hindi | 520 | 61.91 | 57 | 118.00 | 160.86 | 219 | 13 |
| Test | English | 2,507 | 30.31 | 28 | 58.00 | 73.94 | 133 | 7 |
| Test | Hindi | 2,507 | 79.22 | 71 | 158.00 | 206.88 | 293 | 14 |

### What these statistics tell us

- Hindi has longer tokenized sequences on average in this tokenizer’s representation: training mean 49.65 tokens versus 19.74 for English.
- The median is much smaller than the maximum for both languages, indicating a long tail of very long examples.
- For training Hindi, the 95th percentile is 147 tokens, while the 99th percentile is 246 tokens.
- A small number of extreme-length examples can significantly increase padding, activation memory, and attention computation if they are included without a cap.

### Why `max_length = 160`?

The configured maximum sequence length was **160 tokens**. This is a practical coverage-versus-compute trade-off:

- It is above the training P95 for English (54) and Hindi (147), so it retains many common examples.
- It is below the training Hindi P99 (246), so some of the long tail is excluded.
- The filtering check accounts for special tokens: raw token length plus BOS and EOS must fit within the cap.
- Standard attention has a sequence-length-related cost that grows approximately quadratically with sequence length, so bounding length helps control compute and memory.

This choice does **not** mean every sentence fits naturally within 160 tokens. It means the project deliberately traded some long-example coverage for bounded resource requirements.

### Dataset count caveat

Different development notes recorded different training counts: 1,659,083 raw training pairs in the token-length report, 1,652,729 in another summary, and 1,595,098 retained pairs in an earlier pipeline note. These values appear to refer to different stages or runs. **Verify the final preprocessing script and print its actual retained count before publishing a single “final” filtered training count.** The test evaluation count is clearly recorded as 2,394 filtered examples out of 2,507 raw test pairs.

## 5. Tokenizer

The final pipeline uses a **shared Byte-Level Byte-Pair Encoding (BPE) tokenizer** with a vocabulary size of **16,000**.

| Special token | ID | Role |
|---|---:|---|
| `<PAD>` | 0 | Pads sequences to a batch length |
| `<BOS>` | 1 | Marks the beginning of a target sequence |
| `<EOS>` | 2 | Marks the end of a sequence |
| `<UNK>` | 3 | Unknown-token fallback |

### Why use a shared subword tokenizer?

- **Shared vocabulary:** English and Hindi are represented in one token-ID space, simplifying the encoder–decoder model and vocabulary configuration.
- **Subword units:** frequent byte sequences can be merged into useful subword tokens, while uncommon words can be represented as smaller pieces.
- **Byte-level coverage:** byte-level tokenization can represent diverse text and punctuation without needing a separate vocabulary entry for every complete word.
- **Fixed output space:** the decoder predicts over the same configured vocabulary size used by the model.

The project owner reported that tokenizer encode–decode round-trip tests passed. The tokenizer artifact, special-token IDs, pre-tokenizer settings, and decoder configuration must remain consistent between training and inference. A checkpoint trained with one tokenizer should not be evaluated using a newly trained replacement tokenizer.

## 6. Transformer architecture

Both final models use an **encoder–decoder Transformer**. The encoder reads the English source sequence and creates contextual representations. The decoder predicts the Hindi target sequence autoregressively.

### High-level flow

```text
English sentence
      │
      ▼
Shared tokenizer ──► source token IDs
      │
      ▼
Token embeddings + positional information
      │
      ▼
Transformer Encoder (6 blocks)
      │
      ├──────────────────────────────┐
      ▼                              │
Encoder contextual representations   │
                                     │
Hindi target prefix                  │
      │                              │
      ▼                              │
Token embeddings + positional info  │
      │                              │
      ▼                              │
Transformer Decoder (6 blocks) ◄─────┘
  ├─ masked self-attention
  ├─ encoder–decoder cross-attention
  └─ feed-forward network
      │
      ▼
Vocabulary projection → next-token logits
      │
      ▼
Autoregressive decoding / beam search
      │
      ▼
Generated Hindi sentence
```

### Component-by-component explanation

| Component | What it does | Why it is used |
|---|---|---|
| Token embedding | Maps each token ID to a learned vector | Converts discrete IDs into trainable representations |
| Positional encoding | Injects information about token position | Self-attention alone does not inherently encode sequence order |
| Multi-head self-attention | Allows a token representation to combine information from other allowed positions | Models contextual dependencies |
| Encoder self-attention | Source positions attend to source positions, excluding padding | Builds contextual English representations |
| Decoder masked self-attention | Target positions attend only to previous/current target positions | Prevents future-token leakage |
| Cross-attention | Decoder queries attend to encoder representations | Conditions Hindi generation on the English source |
| Feed-forward network | Applies nonlinear transformations independently at each sequence position | Adds model capacity beyond attention |
| Residual connections | Add a block’s input back to its output | Support information flow through deep stacks |
| Layer normalization | Normalizes hidden representations | Helps stabilize optimization |
| Dropout | Randomly drops activations during training | Provides regularization |
| Padding mask | Marks non-padding positions | Prevents the model from attending to padding tokens |
| Vocabulary projection | Maps decoder hidden states to vocabulary-sized logits | Produces next-token scores |

The decoder’s causal mask and its padding mask solve different problems: the causal mask blocks future positions, while the padding mask blocks padded positions. The cross-attention mask corresponds to the source padding mask.

## 7. Final model configurations

The two final models retained the same number of encoder and decoder blocks, but used different hidden dimensions and feed-forward widths.

| Configuration | 23.36M model | 68.73M model |
|---|---:|---:|
| Total parameters | 23,364,224 | 68,732,544 |
| Encoder parameters | 8,835,072 | 27,107,328 |
| Decoder parameters | 14,529,152 | 41,625,216 |
| `d_model` | 256 | 512 |
| Attention heads | 8 | 8 |
| Encoder blocks | 6 | 6 |
| Decoder blocks | 6 | 6 |
| Feed-forward dimension | 1,024 | 2,048 |
| Maximum sequence length | 160 | 160 |
| Dropout | 0.1 | 0.1 |
| Configured learning rate | 0.0003 | 0.0003 |
| Weight decay | 0.01 | 0.01 |
| Gradient clipping norm | 1.0 | 1.0 |
| Batch size | 128 | 64 |
| Gradient accumulation | 2 steps | 2 steps |

### Why change model size?

The smaller configuration uses `d_model = 256` and `ff_dim = 1024`; the larger uses `d_model = 512` and `ff_dim = 2048`. Increasing these dimensions increases representational capacity and parameter count, but also increases memory use and computation per step.

This was an experiment with two complete configurations, not a perfectly controlled parameter-only study: the batch sizes and completed epoch counts differed. As a result, differences in validation loss or BLEU cannot be attributed solely to parameter count.

## 8. Data pipeline and batching

The custom pipeline converts raw bilingual text into model-ready tensors.

1. Read each English–Hindi pair.
2. Tokenize source and target text using the same saved tokenizer.
3. Add BOS/EOS tokens according to the training convention.
4. Reject examples whose tokenized length plus special tokens exceeds the maximum.
5. Pad examples dynamically within each batch.
6. Build masks using `token_id != PAD_ID`.
7. Move source and target tensors to the selected device.

Dynamic padding avoids padding every sentence to 160 tokens when a batch contains only short examples. It reduces unnecessary computation compared with always padding every sample to the global maximum, although actual efficiency depends on how lengths are distributed within each batch.

### Why the masks matter

Padding is a batching convenience, not real language content. If the model attends to padding positions as if they were meaningful tokens, its attention computation can be polluted by artificial inputs. The source mask is used by encoder attention and decoder cross-attention. The target padding mask is used by decoder self-attention, while causal masking prevents access to future target positions.

## 9. Training objective and optimization

### Teacher forcing and shifted targets

During training, the target sequence is shifted to form decoder inputs and labels:

```python
decoder_input = tgt[:, :-1]
labels = tgt[:, 1:]
```

The decoder receives the target prefix and predicts the next token at each position. This is teacher forcing: during training, the correct preceding target tokens are provided rather than feeding back the model’s own predictions.

The loss is vocabulary cross-entropy, ignoring padding:

```python
criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
```

The decoder produces vocabulary logits for each target position. These are reshaped to compare every predicted position with its next-token label.

### Optimization and resource-aware techniques

- **AdamW:** optimizer with configured learning rate `3e-4` and weight decay `0.01`.
- **Mixed precision:** CUDA FP16 autocast and gradient scaling to reduce memory use and potentially accelerate operations.
- **Gradient accumulation:** gradients from two mini-batches are accumulated before an optimizer update.
- **Gradient clipping:** global gradient norm is clipped to `1.0` to limit unusually large updates.
- **Cosine learning-rate scheduling:** learning rate changes over the training run according to the configured scheduler.
- **Validation monitoring:** validation loss is calculated after each epoch.
- **Checkpointing:** model and training states are saved so training can be resumed.
- **Early stopping:** the recorded training loop stops after two consecutive epochs without a new best validation loss.

The reported epoch loss is the mean of per-batch loss values, not necessarily a globally token-weighted average. Keep that convention consistent when comparing runs.

## 10. Implementation from scratch

The project was organized into custom modules rather than relying on a pretrained translation pipeline. The core implementation responsibilities are:

- **Encoder module:** source embeddings/positions and a stack of encoder blocks.
- **Decoder module:** target embeddings/positions, masked self-attention, cross-attention, feed-forward layers, and vocabulary logits.
- **Attention implementation:** query/key/value projections, attention-score calculation, masking, softmax, weighted value aggregation, and output projection.
- **Data pipeline:** tokenization, special tokens, length filtering, padding, and mask generation.
- **Training script:** forward pass, loss, backward pass, gradient accumulation, optimizer/scaler updates, validation, scheduling, and checkpointing.
- **Evaluation script:** checkpoint loading, beam-search decoding, detokenization, and corpus BLEU calculation.

The educational value is in understanding tensor shapes and the data flow between these modules. The final repository should preserve the exact implementation and configuration used to produce the reported checkpoints.

## 11. GPU constraints and FlashAttention

The reported training hardware was an **NVIDIA Tesla T4 GPU**, with approximately 15 GiB of usable VRAM in the environment.

The implementation used standard PyTorch attention operations rather than depending on FlashAttention. This choice was consistent with the project’s goal of implementing the attention calculation explicitly and with the compatibility constraints of the selected software/hardware environment.

A technical nuance is important: the T4 is a Turing GPU with compute capability 7.5. FlashAttention availability depends on the particular version, kernel path, data type, CUDA/PyTorch combination, and GPU architecture. It is too broad to claim that every FlashAttention version is categorically impossible on T4. The accurate project statement is that FlashAttention was not used in this implementation because of the environment’s compatibility constraints and the educational goal of implementing attention directly.

A fused attention kernel could be explored later as an optimization, but it would need to be validated against the current model’s masks, numerical behavior, and hardware.

## 12. Training results

The final experiments took approximately 27 hours for the 23.36M model and 25.6 hours for the 68.73M model, according to the recorded logs.

### Summary table

| Metric | 23.36M model | 68.73M model |
|---|---:|---:|
| Epochs completed | 30 | 15 |
| Best validation loss | 1.4464 (epoch 27) | 1.3864 (epoch 12) |
| Training loss at final recorded epoch | 0.9689 (epoch 29) | 0.6992 (epoch 14) |
| Approximate training time | 27 hours | 25.6 hours |
| Filtered test samples | 2,394 | 2,394 |
| Beam size | 4 | 4 |
| Length penalty | 0.7 | 0.7 |
| Reported corpus BLEU | 6.78 | 7.74 |

### Combined loss curves

![Training and validation loss comparison](results/transformer_loss_comparison.png)

The combined graph lets the reader compare the training and validation trajectories. The 23.36M run continued for 30 epochs; the 68.73M run completed 15 epochs. Because their epoch counts differ, the horizontal extent of the curves should not be interpreted as equal training time.

### Separate loss curves

**23.36M-parameter model**

![23.36M model loss curves](results/loss_23_36M.png)

- Training loss decreased from `2.5124` at epoch 0 to `0.9689` at epoch 29.
- Best validation loss was `1.4464` at epoch 27.
- Validation loss rose slightly at epochs 28 and 29 after the best result.

**68.73M-parameter model**

![68.73M model loss curves](results/loss_68_73M.png)

- Training loss decreased from `2.0342` at epoch 0 to `0.6992` at epoch 14.
- Best validation loss was `1.3864` at epoch 12.
- Validation loss was slightly higher at epochs 13 and 14.

### What can be concluded?

Both models reduced training loss and improved validation loss substantially. The 68.73M model had the lower best validation loss by `0.0600`, and its reported BLEU was higher by `0.96` points. However, the configurations differed in batch size and number of completed epochs, so this does not prove that parameter count alone caused the difference.

The growing gap between training and validation loss late in training is worth noting, but the curves alone do not prove a specific cause such as overfitting.

## 13. Interactive loss curves

A standalone interactive HTML chart is provided separately:

**[Open the interactive loss curves](interactive_transformer_loss_curves.html)**

Hover over or click data points to inspect the exact epoch and loss. The graph marks the best validation epoch for each model.

GitHub README pages do not execute arbitrary interactive HTML inline. Keep the PNG graphs in the README and link to the HTML file, or host the interactive file with GitHub Pages.

## 14. Decoding with beam search

At inference time, the model generates the target sentence autoregressively. It predicts a token, appends it to the prefix, and repeats until EOS or a maximum length is reached.

Greedy decoding chooses the highest-probability next token at each step. Beam search instead retains multiple candidate sequences, allowing later token choices to influence which overall candidate survives. The recorded evaluation used:

- Beam size: `4`
- Length penalty: `0.7`
- Maximum generation length: `160`

Beam search can improve sequence-level selection, but it cannot compensate for incorrect word associations or weak translation patterns learned during training. Beam size and length penalty are inference hyperparameters and should be kept identical for a fair model comparison.

## 15. Evaluation with BLEU

The project used SacreBLEU corpus BLEU to compare generated Hindi translations with reference translations.

BLEU primarily measures n-gram overlap and applies a brevity penalty to discourage overly short translations. It is useful for comparing systems under the same evaluation conditions, but it does not fully measure semantic adequacy, fluency, or grammatical correctness.

### Reported scores

| Model | Corpus BLEU |
|---|---:|
| 23.36M | 6.78 |
| 68.73M | 7.74 |

The 68.73M model scored 0.96 BLEU points higher in the reported evaluation. These values were calculated over **2,394 filtered test examples**, not the entire raw test split of 2,507 pairs. Preserve the SacreBLEU version and full metric signature in the final experiment logs if available.

## 16. Qualitative translation examples

The following examples come from the recorded outputs. They show both occasional reasonable translations and clear failure cases.

### Example A: “A black box in your car?”

**Reference:** `आपकी कार में ब्लैक बॉक्स?`

- **68.73M:** `कार में ब्लैक बक्सा?`
- **23.36M:** `काले बॉक्स में एक काला बॉक्स?`

Both outputs capture the general topic, but the smaller model repeats the concept and neither reproduces the reference naturally.

### Example B: “The tea party is aghast.”

**Reference:** `चाय पार्टी भौचक्की है।`

- **68.73M:** `चाय पार्टी है।`
- **23.36M:** `चाय पार्टी एक चाय है।`

Both fail to preserve the meaning of “aghast.” This illustrates why learning some word correspondences does not guarantee correct sentence-level meaning.

### Example C: “This really is a must for our nation.”

**Reference:** `यह हमारे देश के लिए वास्तव में अनिवार्य वस्तु है।`

- **68.73M:** `यह वास्तव में हमारे देश के लिए एक राष्ट्र होना चाहिए।`
- **23.36M:** `यह वास्तव में हमारे राष्ट्र के लिए आवश्यक है।`

The smaller model gives a more reasonable rendering in this particular example. One example, however, is not enough to establish that it is the better model overall.

### Example D: “There is going to be a change in how we pay these taxes.”

**Reference:** `हमारे द्वारा कर भुगतान किए जाने के तरीके में परिवर्तन होने जा रहा है।`

- **68.73M:** `इन करों में कैसे बदलाव करने के लिए जा रहा है।`
- **23.36M:** `हम इन करों में एक परिवर्तन होने जा रहे हैं।`

Both outputs contain grammatical or semantic errors. These examples are consistent with the low corpus BLEU scores and the observed limitations in generated translations.

## 17. Interpretation and limitations

### What worked

- A complete custom encoder–decoder training and evaluation pipeline was built.
- Both final models showed decreasing training loss and improving validation loss over most of their training.
- The pipeline included mixed precision, gradient accumulation, gradient clipping, checkpointing, and beam-search inference.
- The experiments provided a concrete comparison between two model configurations.

### What did not work well

- Generated translations frequently contain incorrect meaning, awkward grammar, and repeated phrases.
- BLEU scores of 6.78 and 7.74 show limited n-gram overlap under the reported evaluation setup.
- The validation split is small: 520 raw pairs, with an earlier pipeline note reporting 514 retained pairs.
- Different batch sizes and epoch counts mean this is not a perfectly controlled model-scaling experiment.
- Training from random initialization required approximately 25–27 hours for each final run on the reported GPU.
- Earlier experiments affected by a tokenizer issue should not be presented as valid final results.

### Scope of the conclusions

The evidence supports saying that the larger model had lower best validation loss and higher BLEU in these recorded runs. It does not support claiming that the architecture is competitive with pretrained translation models, that more parameters always improve translation, or that a single metric explains all output quality.

## 18. Engineering challenges and lessons learned

1. **Tokenizer integrity is foundational.** A tokenizer mismatch or broken tokenization pipeline can invalidate a large training run.
2. **Architecture correctness is not enough.** Implementing attention correctly does not automatically produce high-quality translation.
3. **Training loss is not translation quality.** A model can predict training tokens increasingly well while still producing weak free-running generations.
4. **Sequence lengths matter.** Long sequences increase compute and memory requirements; length caps trade coverage for tractability.
5. **Masks and target shifting matter.** Padding, causal masking, cross-attention masks, BOS/EOS handling, and shifted labels all affect correctness.
6. **Decoding matters, but has limits.** Beam search changes candidate selection; it does not fix what the model has failed to learn.
7. **Evaluation needs multiple views.** BLEU should be paired with loss curves and qualitative examples.
8. **Compute budgets are part of the experiment.** A 25–27 hour run makes checkpoints, logs, and clear stopping criteria essential.
9. **Document failure honestly.** A learning project can be valuable even when its model is not competitive, provided results and limitations are clearly reported.

## 19. Suggested repository structure

```text
english-hindi-transformer/
├── config.py
├── train.py
├── evaluate.py
├── util/
│   ├── encoder.py
│   ├── decoder.py
│   └── attention.py
├── dataset_processing/
│   ├── shared_tokenizer.py
│   ├── data_pipeline.py
│   └── Dataloaders.py
├── tokenizer/
│   └── shared_tokenizer.json
├── results/
│   ├── transformer_loss_comparison.png
│   ├── loss_23_36M.png
│   ├── loss_68_73M.png
│   └── interactive_transformer_loss_curves.html
└── README.md
```

This is a suggested layout; adjust it to match the actual files in the repository. Do not commit multi-gigabyte datasets or model checkpoints unless there is a specific reason. Provide dataset instructions and checkpoint download instructions where needed.

## 20. Reproducibility checklist

Before publishing, verify:

- [ ] The exact tokenizer artifact and special-token IDs are included.
- [ ] The model configuration used for each checkpoint is recorded.
- [ ] The final filtered training/validation/test counts are printed by the actual pipeline.
- [ ] The training count discrepancy is resolved: 1,659,083 raw pairs, 1,652,729 in one note, and 1,595,098 in earlier pipeline notes.
- [ ] Each BLEU score is linked to the correct checkpoint and evaluation script.
- [ ] Both models use the same filtered test set, beam size, length penalty, and BLEU configuration.
- [ ] The test count is stated as 2,394 filtered examples.
- [ ] The reported BLEU score and SacreBLEU version/signature are preserved.
- [ ] Training logs and loss graphs are included.
- [ ] The README distinguishes measured results from hypotheses and explanations.

## 21. Future improvements

The project was intentionally closed as a learning experiment rather than extended indefinitely. Possible future experiments—if revisited—include:

- Verify final data counts and all preprocessing invariants.
- Compare teacher-forced validation loss with free-running generation diagnostics.
- Add more detailed metrics and human evaluation.
- Run controlled comparisons where batch size, optimizer updates, and data order are held constant.
- Benchmark a compatible fused attention kernel against the explicit PyTorch implementation.
- Compare from-scratch training with fine-tuning a pretrained translation model as a separate experiment.

These are possible directions, not completed work or claims about current performance.

## 22. Conclusion

This project is best understood as an **end-to-end Transformer implementation and model-scaling learning study**. The 68.73M-parameter model achieved the lower recorded validation loss and higher reported BLEU, but both final models remained far below the quality expected of a practical translation system.

The project's contribution is the implementation and investigation of the full pipeline: tokenizer, data preparation, encoder–decoder architecture, training, resource management, decoding, and evaluation. Its results document the gap between building a Transformer correctly and training a strong translation model.
