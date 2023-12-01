import time
import sys
import openvino as ov
import numpy as np
import itertools
from pathlib import Path
from tokenizers import SentencePieceBPETokenizer

sys.path.append("../utils")
from notebook_utils import download_file


base_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1"
model_name = "machine-translation-nar-en-de-0002"
precision = "FP32"
model_base_dir = Path("model")
model_base_dir.mkdir(exist_ok=True)
model_path = model_base_dir / f"{model_name}.xml"
src_tok_dir = model_base_dir / "tokenizer_src"
target_tok_dir = model_base_dir / "tokenizer_tgt"
src_tok_dir.mkdir(exist_ok=True)
target_tok_dir.mkdir(exist_ok=True)

download_file(base_url + f'/{model_name}/{precision}/{model_name}.xml', f"{model_name}.xml", model_base_dir)
download_file(base_url + f'/{model_name}/{precision}/{model_name}.bin', f"{model_name}.bin", model_base_dir)
download_file(f"{base_url}/{model_name}/tokenizer_src/merges.txt", "merges.txt", src_tok_dir)
download_file(f"{base_url}/{model_name}/tokenizer_tgt/merges.txt", "merges.txt", target_tok_dir)
download_file(f"{base_url}/{model_name}/tokenizer_src/vocab.json", "vocab.json", src_tok_dir)
download_file(f"{base_url}/{model_name}/tokenizer_tgt/vocab.json", "vocab.json", target_tok_dir)


core = ov.Core()
model = core.read_model(model_path)
input_name = "tokens"
output_name = "pred"
model.output(output_name)
max_tokens = model.input(input_name).shape[1]

core = ov.Core()


compiled_model = core.compile_model(model, core.available_devices[0])
src_tokenizer = SentencePieceBPETokenizer.from_file(
    str(src_tok_dir / 'vocab.json'),
    str(src_tok_dir / 'merges.txt')
)
tgt_tokenizer = SentencePieceBPETokenizer.from_file(
    str(target_tok_dir / 'vocab.json'),
    str(target_tok_dir / 'merges.txt')
)

def translate(sentence: str) -> str:
    """
    Tokenize the sentence using the downloaded tokenizer and run the model,
    whose output is decoded into a human readable string.

    :param sentence: a string containing the phrase to be translated
    :return: the translated string
    """
    # Remove leading and trailing white spaces
    sentence = sentence.strip()
    assert len(sentence) > 0
    tokens = src_tokenizer.encode(sentence).ids
    # Transform the tokenized sentence into the model's input format
    tokens = [src_tokenizer.token_to_id('<s>')] + \
        tokens + [src_tokenizer.token_to_id('</s>')]
    pad_length = max_tokens - len(tokens)

    # If the sentence size is less than the maximum allowed tokens,
    # fill the remaining tokens with '<pad>'.
    if pad_length > 0:
        tokens = tokens + [src_tokenizer.token_to_id('<pad>')] * pad_length
    assert len(tokens) == max_tokens, "input sentence is too long"
    encoded_sentence = np.array(tokens).reshape(1, -1)

    # Perform inference
    enc_translated = compiled_model({input_name: encoded_sentence})
    output_key = compiled_model.output(output_name)
    enc_translated = enc_translated[output_key][0]

    # Decode the sentence
    sentence = tgt_tokenizer.decode(enc_translated)

    # Remove <pad> tokens, as well as '<s>' and '</s>' tokens which mark the
    # beginning and ending of the sentence.
    for s in ['</s>', '<s>', '<pad>']:
        sentence = sentence.replace(s, '')

    # Transform sentence into lower case and join words by a white space
    sentence = sentence.lower().split()
    sentence = " ".join(key for key, _ in itertools.groupby(sentence))
    return sentence

def run_translator():
    """
    Run the translation in real time, reading the input from the user.
    This function prints the translated sentence and the time
    spent during inference.
    :return:
    """
    while True:
        input_sentence = input()
        if input_sentence == "":
            break

        start_time = time.perf_counter()
        translated = translate(input_sentence)
        end_time = time.perf_counter()
        print(f'Translated: {translated}')
        print(f'Time: {end_time - start_time:.2f}s')

