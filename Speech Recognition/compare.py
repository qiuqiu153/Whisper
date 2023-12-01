#loadModel
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

distil_model_id = "distil-whisper/distil-large-v2"

processor = AutoProcessor.from_pretrained(distil_model_id)

pt_distil_model = AutoModelForSpeechSeq2Seq.from_pretrained(distil_model_id)
pt_distil_model.eval()


#prepare input sample
from datasets import load_dataset

def extract_input_features(sample):
    input_features = processor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
    ).input_features
    return input_features

dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sample = dataset[0]
input_features = extract_input_features(sample)

#Run model inference
import IPython.display as ipd

predicted_ids = pt_distil_model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

display(ipd.Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"]))
print(f"Reference: {sample['text']}")
print(f"Result: {transcription[0]}")



#Load OpenVINO model using Optimum library
from pathlib import Path
from optimum.intel.openvino import OVModelForSpeechSeq2Seq

distil_model_path = Path(distil_model_id.split("/")[-1])

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


#select device
import openvino as ov
import ipywidgets as widgets

core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value="AUTO",
    description="Device:",
    disabled=False,
)

#compile OpenVINO  Model
ov_distil_model.to(device.value)
ov_distil_model.compile()


#Run OpenVINO model inference
predicted_ids = ov_distil_model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

display(ipd.Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"]))
print(f"Reference: {sample['text']}")
print(f"Result: {transcription[0]}")

#Compare performance PyTorch vs OpenVINO
import time
import numpy as np
from tqdm.notebook import tqdm


def measure_perf(model, sample, n=10):
    timers = []
    input_features = extract_input_features(sample)
    for _ in tqdm(range(n), desc="Measuring performance"):
        start = time.perf_counter()
        model.generate(input_features)
        end = time.perf_counter()
        timers.append(end - start)
    return np.median(timers)
perf_distil_torch = measure_perf(pt_distil_model, sample)
perf_distil_ov = measure_perf(ov_distil_model, sample)
print(f"Mean torch {distil_model_path.name} generation time: {perf_distil_torch:.3f}s")
print(f"Mean openvino {distil_model_path.name} generation time: {perf_distil_ov:.3f}s")
print(
    f"Performance {distil_model_path.name} openvino speedup: {perf_distil_torch / perf_distil_ov:.3f}"
)

#Compare with OpenAI Whisper
import gc

model_id = "openai/whisper-large-v2"
model_path = Path(model_id.split("/")[-1])

pt_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
pt_model.eval()

perf_torch = measure_perf(pt_model, sample)

del pt_model
gc.collect()


if not model_path.exists():
    ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_id, export=True, compile=False
    )
    ov_model.half()
    ov_model.generation_config = pt_distil_model.generation_config
    ov_model.save_pretrained(model_path)
else:
    ov_model = OVModelForSpeechSeq2Seq.from_pretrained(model_path, compile=False)

ov_model.to(device.value)
ov_model.compile()

perf_ov = measure_perf(ov_model, sample)
del ov_model
gc.collect()


#last result
print(f"{distil_model_path.name} generation time in PyTorch: {perf_distil_torch:.3f}s")
print(f"{model_path.name} generation time in PyTorch: {perf_torch:.3f}s")
print(
    f"{distil_model_path.name} vs {model_path.name} speedup in PyTorch: {perf_torch / perf_distil_torch:.3f}"
)
print(f"{distil_model_path.name} generation time in OpenVINO: {perf_distil_ov:.3f}s")
print(f"{model_path.name} generation time in OpenVINO: {perf_ov:.3f}s")
print(
    f"{distil_model_path.name} vs {model_path.name} speedup in OpenVINO: {perf_ov / perf_distil_ov:.3f}"
)

#最终结果


# distil-large-v2 generation time in PyTorch: 3.064s
# whisper-large-v2 generation time in PyTorch: 5.054s
# distil-large-v2 vs whisper-large-v2 speedup in PyTorch: 1.649s


# distil-large-v2 generation time in OpenVINO: 1.819s
# whisper-large-v2 generation time in OpenVINO: 3.815s
# distil-large-v2 vs whisper-large-v2 speedup in OpenVINO: 2.097s





