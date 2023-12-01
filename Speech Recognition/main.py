from openVINOPipeline import pipe,prepare_srt
from datasets import load_dataset

if __name__=="__main__":
    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    sample_long = dataset[0]
    result = pipe(sample_long["audio"].copy(), return_timestamps=True)

    srt_lines = prepare_srt(result)

    print("".join(srt_lines))