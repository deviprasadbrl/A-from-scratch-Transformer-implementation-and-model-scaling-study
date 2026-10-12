from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

tokenizer=Tokenizer.from_file("shared_tokenizer.json")
tokenizer.decoder=ByteLevelDecoder()

PAD_ID=tokenizer.token_to_id("<PAD>")
BOS_ID=tokenizer.token_to_id("<BOS>")
EOS_ID=tokenizer.token_to_id("<EOS>")
UNK_ID=tokenizer.token_to_id("<UNK>")

VOCAB_SIZE=tokenizer.get_vocab_size()
