import torch
import torch.nn as nn
import torch.optim as optim
from util.encoder import Encoder
from util.decoder import Decoder
from dataset_processing.data_pipeline import creating_mask,PAD_ID
from dataset_processing.Dataloaders import train_loader,val_loader,test_loader
from config import CONFIG_23M
from paths import LATEST_CHECKPOINT_PATH
import time

config=CONFIG_23M()  #choose this based on the model

RESUME=False
checkpoint_path=LATEST_CHECKPOINT_PATH

device=torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
print(f"Running on device: {device}")

encoder_model=Encoder(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_blocks=config.encoder_blocks,
        vocab_size=config.vocab_size,
        max_length=config.max_length,
        ff_dim=config.ff_dim,
        dropout_rate=config.dropout_rate
).to(device)

decoder_model=Decoder(
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_blocks=config.decoder_blocks,
        vocab_size=config.vocab_size,
        max_length=config.max_length,
        ff_dim=config.ff_dim,
        dropout_rate=config.dropout_rate
).to(device)

criterion=nn.CrossEntropyLoss(ignore_index=PAD_ID)
all_params=list(encoder_model.parameters())+list(decoder_model.parameters())
optimizer=optim.AdamW(all_params,lr=config.lr,weight_decay=config.weight_decay)
scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.total_epochs)

scaler=torch.amp.GradScaler("cuda",enabled=(device.type=="cuda"))

if RESUME:

    best_translation_model=torch.load(
        checkpoint_path,
        map_location=device
    )

    encoder_model.load_state_dict(best_translation_model["encoder_state_dict"])
    decoder_model.load_state_dict(best_translation_model["decoder_state_dict"])

    optimizer.load_state_dict(best_translation_model["optimizer_state_dict"])
    scheduler.load_state_dict(best_translation_model["scheduler_state_dict"])
    scaler.load_state_dict(best_translation_model["scaler_state_dict"])

    best_val=best_translation_model["best_val_loss"]
    patience_counter=best_translation_model["patience_counter"]
    start_epoch=best_translation_model["epoch"]+1

else:
    best_val=float("inf")
    patience_counter=0
    start_epoch=0

patience=2

gradient_accumulation_steps=config.gradient_acculumation


for epoch in range(start_epoch,config.total_epochs):

    epoch_start=time.time()

    encoder_model.train()
    decoder_model.train()

    epoch_loss=0.0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader):

        src=batch["src"].to(device)
        tgt=batch["tgt"].to(device)

        decoder_input =tgt[:, :-1]
        labels=tgt[:, 1:]

        with torch.amp.autocast(device_type="cuda",dtype=torch.float16,enabled=(device.type=="cuda")):

            encoder_mask=creating_mask(src)
            decoder_mask=creating_mask(decoder_input)
            cross_mask=encoder_mask

            encoder_out=encoder_model(src,encoder_mask)

            decoder_out=decoder_model(decoder_input,context=encoder_out,self_mask=decoder_mask,cross_mask=cross_mask)

            loss=criterion(decoder_out.reshape(-1, config.vocab_size),labels.reshape(-1))

            epoch_loss+=loss.item()

            accumulation_loss=(loss/gradient_accumulation_steps)

        scaler.scale(accumulation_loss).backward()


        if ((step + 1)%gradient_accumulation_steps==0 or (step + 1)==len(train_loader)):

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(all_params,config.max_norm)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)


        if (step + 1)%500==0:

            elapsed=time.time()-epoch_start

            batches_done=step+1
            batches_total=len(train_loader)

            time_per_batch=elapsed/batches_done

            remaining=(batches_total-batches_done)*time_per_batch

            print(
                f"Epoch {epoch} | "
                f"Batch {batches_done}/{batches_total} | "
                f"Loss {loss.item():.4f} | "
                f"ETA {remaining / 60:.1f} min"
            )




    mean_train_loss=(epoch_loss/len(train_loader))

    print(
        f"Epoch: {epoch} "
        f"Train Loss: {mean_train_loss:.4f}"
    )

    encoder_model.eval()
    decoder_model.eval()

    val_epoch_loss=0.0

    with torch.no_grad():

        for batch in val_loader:

            src=batch["src"].to(device)
            tgt=batch["tgt"].to(device)

            decoder_input=tgt[:, :-1]
            labels=tgt[:, 1:]

            encoder_mask=creating_mask(src)
            decoder_mask=creating_mask(decoder_input)
            cross_mask=encoder_mask

            with torch.amp.autocast(device_type="cuda",dtype=torch.float16,enabled=(device.type=="cuda")):

                encoder_out=encoder_model(src,encoder_mask)

                decoder_out=decoder_model(decoder_input,context=encoder_out,self_mask=decoder_mask,cross_mask=cross_mask)

                val_loss=criterion(decoder_out.reshape(-1, config.vocab_size),labels.reshape(-1))

                val_epoch_loss+=val_loss.item()


    mean_loss=(val_epoch_loss / len(val_loader))

    print(
        f"Epoch: {epoch} "
        f"Val Loss: {mean_loss:.4f}"
    )


    if mean_loss<best_val:

        best_val=mean_loss
        patience_counter=0

        checkpoint={
                    "epoch": epoch,
                    "encoder_state_dict": encoder_model.state_dict(),
                    "decoder_state_dict": decoder_model.state_dict(),
                    "best_val_loss": best_val
                }

        torch.save(checkpoint,"best_translation_model.pt")
                

        print(f"Best model saved at epoch: {epoch}")

    else:
        patience_counter+=1

        print(f"The model didn't beat the best val "f"({patience_counter}/{patience})")


    scheduler.step()
    print(f"Learning rate: "f"{optimizer.param_groups[0]['lr']:.8f}")
    latest_checkpoint={
                        "epoch": epoch,
                        "encoder_state_dict": encoder_model.state_dict(),
                        "decoder_state_dict": decoder_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_val_loss": best_val,
                        "scheduler_state_dict": scheduler.state_dict(),
                        "patience_counter":patience_counter
                        }
    torch.save(latest_checkpoint,"latest_translation_model.pt")
    print(f"Latest model saved at epoch: {epoch}")

    epoch_time=time.time()-epoch_start

    print(f"Epoch {epoch} Time: "f"{epoch_time / 60:.2f} minutes")

    if patience_counter>=patience:

        print("EARLY STOPPING TRIGGERED")

        epoch_time=time.time()-epoch_start

        print(f"Epoch {epoch} Time: "f"{epoch_time / 60:.2f} minutes")
        break