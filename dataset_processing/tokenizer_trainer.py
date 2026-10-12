from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from datasets import load_from_disk
from paths import DATASET_PATH


dataset=load_from_disk(DATASET_PATH)


tokenizer=Tokenizer(BPE(unk_token="<UNK>"))


tokenizer.pre_tokenizer=ByteLevel(
    add_prefix_space=False
)


special_tokens=[
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>"
]


trainer=BpeTrainer(
    vocab_size=16000,
    min_frequency=2,
    special_tokens=special_tokens,
    initial_alphabet=ByteLevel.alphabet()
)


def corpus_iterator():
    for example in dataset["train"]:
        yield example["translation"]["en"]
        yield example["translation"]["hi"]


tokenizer.train_from_iterator(
    corpus_iterator(),
    trainer=trainer
)


tokenizer.decoder=ByteLevelDecoder()


tokenizer.save("shared_tokenizer.json")
