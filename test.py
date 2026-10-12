import torch
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from util.encoder import Encoder
from util.decoder import Decoder

from dataset_processing.data_pipeline import creating_mask
from dataset_processing.shared_tokenizer import tokenizer,BOS_ID,EOS_ID,PAD_ID

from dataset_processing.Dataloaders import test_loader
from config import Config_23
from sacrebleu.metrics import BLEU
from paths import BEST_CHECKPOINT_PATH


config=Config_23()

BEAM_SIZE=4
LENGTH_PENALTY=0.7

CHECKPOINT_PATH=BEST_CHECKPOINT_PATH




device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Running on device: {device}")

tokenizer.decoder = ByteLevelDecoder()

test_dataset=test_loader.dataset

print(f"Filtered test samples: {len(test_dataset)}")
print(f"Test batches: {len(test_loader)}")


checkpoint=torch.load(CHECKPOINT_PATH,map_location=device,weights_only=False)

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")


encoder_model=Encoder(
    d_model=config.d_model,
    num_heads=config.num_heads,
    num_blocks=config.encoder_blocks,
    vocab_size=config.vocab_size,
    max_length=config.max_length,
    ff_dim=config.ff_dim,
    dropout_rate=config.dropout_rate,
).to(device)

decoder_model=Decoder(
    d_model=config.d_model,
    num_heads=config.num_heads,
    num_blocks=config.decoder_blocks,
    vocab_size=config.vocab_size,
    max_length=config.max_length,
    ff_dim=config.ff_dim,
    dropout_rate=config.dropout_rate,
).to(device)



encoder_model.load_state_dict(checkpoint["encoder_state_dict"])

decoder_model.load_state_dict(checkpoint["decoder_state_dict"])

encoder_model.eval()
decoder_model.eval()

print("Model weights loaded successfully.")


def beam_search(
    src,
    encoder_out,
    encoder_mask,
    beam_size=4,
    max_length=None,
    length_penalty=0.7,
):

    if max_length is None:
        max_length=config.max_length

    batch_size = src.size(0)

    encoder_out=(encoder_out.unsqueeze(1).expand(batch_size,beam_size,encoder_out.size(1),encoder_out.size(2)).reshape(batch_size * beam_size,encoder_out.size(1),encoder_out.size(2),))

    encoder_mask=(
        encoder_mask
        .unsqueeze(1)
        .expand(
            batch_size,
            beam_size,
            encoder_mask.size(1),
        )
        .reshape(
            batch_size * beam_size,
            encoder_mask.size(1),
        )
    )

    # Initialize sequences with BOS
    sequences=torch.full((batch_size * beam_size, 1),BOS_ID,dtype=torch.long,device=device)

    beam_scores=torch.full((batch_size, beam_size),float("-inf"),device=device)

    beam_scores[:, 0]=0.0
    beam_scores=beam_scores.reshape(-1)

    finished=torch.zeros(batch_size * beam_size,dtype=torch.bool,device=device)

    for step in range(max_length - 1):

        decoder_mask=creating_mask(sequences)

        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=(device.type == "cuda")):

            decoder_out=decoder_model(sequences,context=encoder_out,self_mask=decoder_mask,cross_mask=encoder_mask)

        logits=decoder_out[:, -1, :]

        log_probs=torch.log_softmax(logits,dim=-1)

        log_probs[finished]=float("-inf")
        log_probs[finished,EOS_ID]=0.0

        vocab_size=log_probs.size(-1)

        candidate_scores=(beam_scores.unsqueeze(1)+log_probs)

        candidate_scores=candidate_scores.reshape(batch_size,beam_size * vocab_size,)

        top_scores,top_indices=torch.topk(candidate_scores,beam_size,dim=-1,)

        next_beam=top_indices//vocab_size
        next_tokens=top_indices%vocab_size

        batch_offset=(torch.arange(batch_size,device=device)*beam_size).unsqueeze(1)

        gather_indices=(batch_offset + next_beam).reshape(-1)

        sequences=sequences[gather_indices]
        encoder_out=encoder_out[gather_indices]
        encoder_mask=encoder_mask[gather_indices]
        finished=finished[gather_indices]

        next_tokens=next_tokens.reshape(-1)

        sequences=torch.cat([sequences,next_tokens.unsqueeze(1),],dim=1,)

        beam_scores=top_scores.reshape(-1)

        finished=finished|(next_tokens==EOS_ID)

        if finished.all():
            break


    lengths=(sequences != PAD_ID).sum(dim=1).float() - 1
    lengths=lengths.clamp(min=1)

    normalized_scores=(beam_scores/(lengths ** length_penalty))

    normalized_scores=normalized_scores.reshape(batch_size,beam_size)


    best_indices=torch.argmax(normalized_scores,dim=1)

    sequences=sequences.reshape(batch_size,beam_size-1)

    batch_indices=torch.arange(batch_size,device=device)

    best_sequences=sequences[batch_indices,best_indices]

    return best_sequences


def decode_ids(ids):

    tokens=[]

    for token_id in ids:

        token_id=token_id.item()

        if token_id==BOS_ID:
            continue

        if token_id==EOS_ID:
            break

        if token_id==PAD_ID:
            continue

        tokens.append(token_id)

    return tokenizer.decode(tokens)


predictions=[]
references=[]

total_samples=0

with torch.inference_mode():

    for batch_idx, batch in enumerate(test_loader):

        src=batch["src"].to(device)
        tgt=batch["tgt"].to(device)


        encoder_mask=creating_mask(src)

        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=(device.type == "cuda")):

            encoder_out=encoder_model(src,encoder_mask)


        predicted_ids=beam_search(
            src=src,
            encoder_out=encoder_out,
            encoder_mask=encoder_mask,
            beam_size=BEAM_SIZE,
            max_length=config.max_length,
            length_penalty=LENGTH_PENALTY,
        )

        batch_size=src.size(0)

        for i in range(batch_size):

            english=decode_ids(src[i])
            reference=decode_ids(tgt[i])
            prediction=decode_ids(predicted_ids[i])

            predictions.append(prediction)
            references.append(reference)

            if batch_idx==0 and i<10:

                print("\n" + "=" *70)
                print(f"Sample {i + 1}")

                print("\nEnglish:")
                print(english)

                print("\nReference Hindi:")
                print(reference)

                print(f"\nBeam Search (beam={BEAM_SIZE}):")
                print(prediction)

                print("=" * 70)

        total_samples+=batch_size

        # Progress update
        if (batch_idx + 1)%10==0:

            print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")



assert len(predictions)==len(references), (
    f"Prediction count ({len(predictions)}) does not match "
    f"reference count ({len(references)})"
)

assert len(references)==len(test_dataset), (
    f"Evaluated {len(references)} samples, but the test "
    f"dataset contains {len(test_dataset)} samples. "
    "Check that test_loader uses shuffle=False."
)

print(f"\nTotal evaluated samples: {total_samples}")



bleu=BLEU()

bleu_score=bleu.corpus_score(predictions,[references],)



print("\n")
print("=" * 70)
print("FINAL BEAM SEARCH TEST RESULTS")
print("=" * 70)

print(f"BLEU Score      : {bleu_score.score / 100:.4f}")
print(f"BLEU Score x100 : {bleu_score.score:.2f}")
print(f"Test Samples    : {total_samples}")
print(f"Beam Size       : {BEAM_SIZE}")
print(f"Length Penalty  : {LENGTH_PENALTY}")
print(f"Checkpoint Epoch: {checkpoint['epoch']}")
print("=" * 70)
