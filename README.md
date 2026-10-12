# English–Hindi Neural Machine Translation from Scratch
## A Transformer implementation and model-scaling study

This project implements an encoder–decoder Transformer in PyTorch for English-to-Hindi translation, trained from randomly initialized weights. It includes custom attention, encoder/decoder blocks, a shared Byte-Level BPE tokenizer, data processing, mixed-precision training, checkpointing, beam search, and SacreBLEU evaluation.

## Final results

| Metric | 23.36M model | 68.73M model |
|---|---:|---:|
| Parameters | 23,364,224 | 68,732,544 |
| Best validation loss | 1.4464 (epoch 27) | 1.3864 (epoch 12) |
| Final recorded train loss | 0.9689 | 0.6992 |
| Approximate training time | 27 h | 25.6 h |
| Filtered test examples | 2,394 | 2,394 |
| Beam size / length penalty | 4 / 0.7 | 4 / 0.7 |
| SacreBLEU corpus BLEU | 6.78 | 7.74 |

![Combined loss curves](transformer_loss_comparison.png)

## Dataset and sequence length

The uploaded token-length report lists 1,659,083 raw training pairs, 520 validation pairs, and 2,507 test pairs.

| Split | Language | Sentences | Mean | Median | P95 | P99 | Max | Min |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Train | English | 1,659,083 | 19.74 | 14 | 54.0 | 92.00 | 3,228 | 2 |
| Train | Hindi | 1,659,083 | 49.65 | 33 | 147.0 | 246.00 | 5,337 | 2 |
| Validation | English | 520 | 26.44 | 24 | 50.0 | 67.81 | 105 | 8 |
| Validation | Hindi | 520 | 61.91 | 57 | 118.0 | 160.86 | 219 | 13 |
| Test | English | 2,507 | 30.31 | 28 | 58.0 | 73.94 | 133 | 7 |
| Test | Hindi | 2,507 | 79.22 | 71 | 158.0 | 206.88 | 293 | 14 |

The 160-token cap is a compute/coverage trade-off: it exceeds training P95 for English and Hindi, but remains below Hindi P99. BOS/EOS are included in filtering. The test loader retained 2,394 of 2,507 raw test pairs. Reconcile training counts before publication: the notes contain 1,659,083 raw pairs, 1,652,729 examples in one summary, and 1,595,098 retained pairs in earlier pipeline notes.

## Architecture and training

Both models use six encoder blocks, six decoder blocks, eight heads, max length 160, and dropout 0.1. The smaller model uses d_model=256 and FFN=1,024; the larger uses d_model=512 and FFN=2,048. Shared tokenizer vocabulary size is 16,000 with PAD=0, BOS=1, EOS=2, UNK=3.

Training uses teacher forcing (`decoder_input = tgt[:, :-1]`, `labels = tgt[:, 1:]`), cross-entropy ignoring PAD, AdamW (lr=3e-4, weight decay=0.01), CUDA mixed precision, gradient accumulation of 2, gradient clipping at 1.0, cosine scheduling, checkpointing, and validation monitoring.

The reported GPU was an NVIDIA Tesla T4 (Turing, compute capability 7.5). Standard PyTorch attention operations were used to keep the implementation explicit and educational. FlashAttention support depends on version, architecture, data type, and kernel path; state that it was not used because of the chosen environment's compatibility constraints and the from-scratch objective, rather than claiming universal incompatibility on T4.

## Evaluation and limitations

Beam search used beam size 4 and length penalty 0.7. Corpus BLEU is based on 2,394 filtered examples, not the full raw test split. BLEU measures n-gram overlap and does not fully capture semantic adequacy or fluency.

Generated translations contain repetition, mistranslation, and grammatical errors. Validation loss improved for both models, but this does not guarantee strong generated text. The 68.73M model had lower best validation loss and higher BLEU by 0.96 points, but the experiments differed in batch size and epoch count, so parameter count is not the only variable.

## Lessons learned

- Tokenizer and preprocessing integrity are foundational.
- Lower training loss does not guarantee good translation.
- Parameter count alone does not determine quality.
- Masking, target shifting, padding, EOS handling, and decoding matter.
- Quantitative metrics should be paired with qualitative examples.
- Document limitations honestly; this is a learning project, not a production translator.
