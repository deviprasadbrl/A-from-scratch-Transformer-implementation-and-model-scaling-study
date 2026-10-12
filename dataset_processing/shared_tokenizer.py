from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from paths import TOKENIZER_PATH

tokenizer=Tokenizer.from_file(TOKENIZER_PATH)
tokenizer.decoder=ByteLevelDecoder()

PAD_ID=tokenizer.token_to_id("<PAD>")
BOS_ID=tokenizer.token_to_id("<BOS>")
EOS_ID=tokenizer.token_to_id("<EOS>")
UNK_ID=tokenizer.token_to_id("<UNK>")

VOCAB_SIZE=tokenizer.get_vocab_size()
