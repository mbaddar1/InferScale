import sys

import transformers
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizerFast, BertTokenizer

print(transformers.__version__)
if __name__=="__main__":
    # core code
    # Load a pre-trained BERT tokenizer
    # Direct instantiation skips AutoTokenizer's "Fast-by-default" logic
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(f"Is fast? : {tokenizer.is_fast}")  # This should now be False
    print("==============================")
    sys.exit(-1)
    #
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased',use_fast=False,force_download=True)
    # Verify the tokenizer loaded correctly
    assert tokenizer is not None, "Tokenizer failed to load"
    assert tokenizer.model_max_length == 512, f"Expected max length 512, got {tokenizer.model_max_length}"
    print(f"Tokenizer type: {type(tokenizer).__name__}")
    print(f"Is fast ? : {tokenizer.is_fast}")
    # print(f"Model max length: {tokenizer.model_max_length}")
    # print(f"Vocabulary size: {len(tokenizer.get_vocab())}")
    # print(f"Padding token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    # print(f"Unknown token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
