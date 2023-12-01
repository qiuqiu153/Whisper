from transformers import pipeline
from init import*
from datasets import load_dataset

from pathlib import Path
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoModelForSpeechSeq2Seq
import gc
from util import quantize

if not distil_model_path.exists():
    ov_distil_model = OVModelForSpeechSeq2Seq.from_pretrained(
        distil_model_id, export=True, compile=False
    )
    ov_distil_model.half()
    ov_distil_model.save_pretrained(distil_model_path)
else:
    ov_distil_model = OVModelForSpeechSeq2Seq.from_pretrained(
        distil_model_path, compile=False
    )

# pt_distil_model = AutoModelForSpeechSeq2Seq.from_pretrained(distil_model_id)
# pt_distil_model.eval()
# ov_distil_model.generation_config = pt_distil_model.generation_config
# del pt_distil_model
# gc.collect()

#quantize
ov_distil_model=quantize(ov_distil_model)
gc.collect()

pipe = pipeline(
    "automatic-speech-recognition",
    model=ov_distil_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
)




def format_timestamp(seconds: float):
    """
    format time in srt-file expected format
    """
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return (
        f"{hours}:" if hours > 0 else "00:"
    ) + f"{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def prepare_srt(transcription):
    """
    Format transcription into srt file format
    """
    segment_lines = []
    for idx, segment in enumerate(transcription["chunks"]):
        segment_lines.append(str(idx + 1) + "\n")
        timestamps = segment["timestamp"]
        time_start = format_timestamp(timestamps[0])
        time_end = format_timestamp(timestamps[1])
        time_str = f"{time_start} --> {time_end}\n"
        segment_lines.append(time_str)
        segment_lines.append(segment["text"] + "\n\n")
    return segment_lines

