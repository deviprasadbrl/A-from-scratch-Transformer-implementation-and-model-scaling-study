# English–Hindi Transformer from Scratch

An educational English-to-Hindi neural machine translation project implemented in PyTorch. The encoder–decoder Transformer, data pipeline, training loop, decoding, and evaluation were built for learning and experimentation.

## Project overview

- **Task:** English → Hindi translation
- **Dataset:** IIT Bombay English–Hindi parallel corpus
- **Tokenizer:** Shared Byte-Level BPE, vocabulary size 16,000
- **Hardware:** NVIDIA Tesla T4
- **Approach:** Train from random initialization; teacher forcing; AdamW; mixed precision; gradient accumulation; gradient clipping; checkpointing

## Dataset and sequence length

The token-length report lists 1,659,083 raw training pairs, 520 validation pairs, and 2,507 test pairs before filtering. Lengths below are tokenizer tokens, not words.

| Split | Language | Sentences | Mean | Median | P95 | P99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| Train | English | 1,659,083 | 19.74 | 14 | 54.00 | 92.00 | 3,228 |
| Train | Hindi | 1,659,083 | 49.65 | 33 | 147.00 | 246.00 | 5,337 |
| Validation | English | 520 | 26.44 | 24 | 50.00 | 67.81 | 105 |
| Validation | Hindi | 520 | 61.91 | 57 | 118.00 | 160.86 | 219 |
| Test | English | 2,507 | 30.31 | 28 | 58.00 | 73.94 | 133 |
| Test | Hindi | 2,507 | 79.22 | 71 | 158.00 | 206.88 | 293 |

`max_length = 160` was chosen as a compute/coverage trade-off. It exceeds the training P95 for Hindi (147) but is below Hindi P99 (246). BOS/EOS tokens count toward the limit. The test loader retained 2,394 of 2,507 raw test pairs.

## Architecture

Both models use six encoder blocks, six decoder blocks, and eight attention heads. The encoder builds contextual source representations; the decoder uses masked self-attention and cross-attention to generate Hindi tokens autoregressively. Padding masks block PAD positions and causal masking blocks future target tokens.

The shared tokenizer uses `<PAD>=0`, `<BOS>=1`, `<EOS>=2`, and `<UNK>=3`. Attention and Transformer components were implemented using standard PyTorch operations. FlashAttention was not used due to the chosen environment's compatibility constraints and the goal of explicitly implementing attention; support depends on software version, kernel, dtype, and GPU architecture.

## Training and model configurations

Training uses shifted targets (`decoder_input = tgt[:, :-1]`, `labels = tgt[:, 1:]`), cross-entropy with PAD ignored, AdamW (`lr=3e-4`, weight decay `0.01`), mixed precision, gradient accumulation, gradient clipping (`1.0`), cosine scheduling, validation monitoring, and checkpoints.

| Metric | 23.36M Transformer | 68.73M Transformer |
|---|---:|---:|
| `d_model` | 256 | 512 |
| FFN dimension | 1,024 | 2,048 |
| Encoder / decoder blocks | 6 / 6 | 6 / 6 |
| Batch size | 128 | 64 |
| Epochs completed | 30 | 15 |
| Best validation loss | 1.4464 (epoch 27) | 1.3864 (epoch 12) |
| Approx. training time | 27 hours | 25.6 hours |
| Filtered test examples | 2,394 | 2,394 |
| Beam size / length penalty | 4 / 0.7 | 4 / 0.7 |
| Reported corpus BLEU | 6.78 | 7.74 |

## Loss curves

All graphs below use the **actual recorded training and validation losses from these two Transformer runs only**.

### Combined comparison

![Combined Transformer loss curves](transformer_loss_comparison.png)

[Download combined graph](transformer_loss_comparison.png)

### 23.36M-parameter Transformer

![23.36M Transformer loss curves](loss_23_36M.png)

[Download 23.36M graph](loss_23_36M.png)

### 68.73M-parameter Transformer

![68.73M Transformer loss curves](loss_68_73M.png)

[Download 68.73M graph](loss_68_73M.png)

## Sample outputs

**English:** A black box in your car?  
**Reference:** आपकी कार में ब्लैक बॉक्स?  
**68.73M:** कार में ब्लैक बक्सा?  
**23.36M:** काले बॉक्स में एक काला बॉक्स?

**English:** The tea party is aghast.  
**Reference:** चाय पार्टी भौचक्की है।  
**68.73M:** चाय पार्टी है।  
**23.36M:** चाय पार्टी एक चाय है।

## Results and limitations

The 68.73M model reached a best validation loss 0.0600 lower and a reported BLEU 0.96 points higher than the 23.36M model. This was not a parameter-only controlled experiment because batch sizes and completed epochs differed. Both models produce grammatical and semantic errors and are not production-ready translators. BLEU was calculated on 2,394 filtered test examples.

A few development notes contain different training-set counts across pipeline stages; verify the final preprocessing output before claiming a single filtered training count. This project is presented as a from-scratch Transformer learning project, emphasizing implementation, experimentation, and honest evaluation.
