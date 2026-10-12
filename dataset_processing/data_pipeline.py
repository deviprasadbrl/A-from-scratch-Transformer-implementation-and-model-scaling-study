import torch
from torch.nn.utils.rnn import pad_sequence
from dataset_processing.shared_tokenizer import tokenizer,BOS_ID,EOS_ID,PAD_ID
from torch.utils.data import Dataset
from config import Config


def filter_by_length(sample):
    src_ids=tokenize_sentence(sample["translation"]["en"])
    tgt_ids=tokenize_sentence(sample["translation"]["hi"])

    return len(src_ids)<=Config.max_length and len(tgt_ids)<=Config.max_length


def tokenize_sentence(text):
    ids=tokenizer.encode(text).ids

    ids=[BOS_ID]+ids+[EOS_ID]
    assert ids[0]==BOS_ID
    assert ids[-1]==EOS_ID

    return ids

def collate_fn(batch):
        src_batch=[torch.tensor(item["src_ids"], dtype=torch.long) for item in batch]

        tgt_batch=[torch.tensor(item["tgt_ids"], dtype=torch.long)for item in batch]

        src_batch=pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=PAD_ID
         )

        tgt_batch=pad_sequence(
            tgt_batch,
            batch_first=True,
            padding_value=PAD_ID
        )

        return{
            "src": src_batch,
            "tgt": tgt_batch
        }


def creating_mask(x):
     return x!=PAD_ID

class TranslationDataset(Dataset):

    def __init__(self, dataset):
        self.dataset=dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):

        sample=self.dataset[idx]["translation"]

        src_text=sample["en"]
        tgt_text=sample["hi"]

        src_ids=tokenize_sentence(src_text)
        tgt_ids=tokenize_sentence(tgt_text)

        return {
            "src_ids":src_ids,
            "tgt_ids":tgt_ids
        }