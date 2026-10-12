from torch.utils.data import DataLoader
from dataset_processing.data_pipeline import TranslationDataset,collate_fn,filter_by_length,tokenize_sentence
from datasets import load_from_disk
from config import Config_23M
from paths import DATASET_PATH

BATCH_SIZE=Config_23M.batch_size

dataset=load_from_disk(DATASET_PATH)

dataset=dataset.filter(filter_by_length)

train_dataset=TranslationDataset(dataset["train"])
test_dataset=TranslationDataset(dataset["test"])
val_dataset=TranslationDataset(dataset["validation"])

train_loader=DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

test_loader=DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

val_loader=DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)


