to_quantize = widgets.Checkbox(
    value=True,
    description='Quantization',
    disabled=False,
)
import urllib.request
urllib.request.urlretrieve(
    url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/skip_kernel_extension.py',
    filename='skip_kernel_extension.py'
)

from itertools import islice
from typing import List, Any
from openvino import Tensor

class InferRequestWrapper:
    def __init__(self, request, data_cache: List):
        self.request = request
        self.data_cache = data_cache

    def __call__(self, *args, **kwargs):
        self.data_cache.append(*args)
        return self.request(*args, *kwargs)

    def infer(self, inputs: Any = None, shared_memory: bool = False):
        self.data_cache.append(inputs)
        return self.request.infer(inputs, shared_memory)

    def start_async(
        self,
        inputs: Any = None,
        userdata: Any = None,
        shared_memory: bool = False,
    ):
        self.data_cache.append(inputs)
        self.request.infer(inputs, shared_memory)

    def wait(self):
        pass

    def get_tensor(self, name: str):
        return Tensor(self.request.results[name])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.request, attr)
def collect_calibration_dataset(ov_model, calibration_dataset_size):
    # Overwrite model request properties, saving the original ones for restoring later
    original_encoder_request = ov_model.encoder.request
    original_decoder_with_past_request = ov_model.decoder_with_past.request
    encoder_calibration_data = []
    decoder_calibration_data = []
    ov_model.encoder.request = InferRequestWrapper(original_encoder_request, encoder_calibration_data)
    ov_model.decoder_with_past.request = InferRequestWrapper(original_decoder_with_past_request,
                                                             decoder_calibration_data)

    calibration_dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    for sample in tqdm(islice(calibration_dataset, calibration_dataset_size), desc="Collecting calibration data",
                       total=calibration_dataset_size):
        input_features = extract_input_features(sample)
        ov_model.generate(input_features)

    ov_model.encoder.request = original_encoder_request
    ov_model.decoder_with_past.request = original_decoder_with_past_request

    return encoder_calibration_data, decoder_calibration_data

import shutil
import nncf

CALIBRATION_DATASET_SIZE = 10
quantized_distil_model_path = Path(f"{distil_model_path}_quantized")


def quantize(ov_model, calibration_dataset_size):
    if not quantized_distil_model_path.exists():
        encoder_calibration_data, decoder_calibration_data = collect_calibration_dataset(
            ov_model, calibration_dataset_size
        )
        print("Quantizing encoder")
        quantized_encoder = nncf.quantize(
            ov_model.encoder.model,
            nncf.Dataset(encoder_calibration_data),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(encoder_calibration_data),
            model_type=nncf.ModelType.TRANSFORMER,
            # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.50)
        )
        ov.save_model(quantized_encoder, quantized_distil_model_path / "openvino_encoder_model.xml")
        del quantized_encoder
        del encoder_calibration_data
        gc.collect()

        print("Quantizing decoder with past")
        quantized_decoder_with_past = nncf.quantize(
            ov_model.decoder_with_past.model,
            nncf.Dataset(decoder_calibration_data),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(decoder_calibration_data),
            model_type=nncf.ModelType.TRANSFORMER,
            # Smooth Quant algorithm reduces activation quantization error; optimal alpha value was obtained through grid search
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=0.95)
        )
        ov.save_model(quantized_decoder_with_past, quantized_distil_model_path / "openvino_decoder_with_past_model.xml")
        del quantized_decoder_with_past
        del decoder_calibration_data
        gc.collect()

        # Copy the config file and the first-step-decoder manually
        shutil.copy(distil_model_path / "config.json", quantized_distil_model_path / "config.json")
        shutil.copy(distil_model_path / "openvino_decoder_model.xml", quantized_distil_model_path / "openvino_decoder_model.xml")
        shutil.copy(distil_model_path / "openvino_decoder_model.bin", quantized_distil_model_path / "openvino_decoder_model.bin")

    quantized_ov_model = OVModelForSpeechSeq2Seq.from_pretrained(quantized_distil_model_path, compile=False)
    quantized_ov_model.to(device.value)
    quantized_ov_model.compile()
    return quantized_ov_model


ov_quantized_distil_model = quantize(ov_distil_model, CALIBRATION_DATASET_SIZE)


#Run quantized model inference
dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
sample = dataset[0]
input_features = extract_input_features(sample)

predicted_ids = ov_distil_model.generate(input_features)
transcription_original = processor.batch_decode(predicted_ids, skip_special_tokens=True)

predicted_ids = ov_quantized_distil_model.generate(input_features)
transcription_quantized = processor.batch_decode(predicted_ids, skip_special_tokens=True)

display(ipd.Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"]))
print(f"Original : {transcription_original[0]}")
print(f"Quantized: {transcription_quantized[0]}")

import time
from contextlib import contextmanager
from jiwer import wer, wer_standardize


TEST_DATASET_SIZE = 50
MEASURE_TIME = False

@contextmanager
def time_measurement():
    global MEASURE_TIME
    try:
        MEASURE_TIME = True
        yield
    finally:
        MEASURE_TIME = False

def time_fn(obj, fn_name, time_list):
    original_fn = getattr(obj, fn_name)

    def wrapper(*args, **kwargs):
        if not MEASURE_TIME:
            return original_fn(*args, **kwargs)
        start_time = time.perf_counter()
        result = original_fn(*args, **kwargs)
        end_time = time.perf_counter()
        time_list.append(end_time - start_time)
        return result

    setattr(obj, fn_name, wrapper)

def calculate_transcription_time_and_accuracy(ov_model, test_samples):
    encoder_infer_times = []
    decoder_with_past_infer_times = []
    whole_infer_times = []
    time_fn(ov_model, "generate", whole_infer_times)
    time_fn(ov_model.encoder, "forward", encoder_infer_times)
    time_fn(ov_model.decoder_with_past, "forward", decoder_with_past_infer_times)

    ground_truths = []
    predictions = []
    for data_item in tqdm(test_samples, desc="Measuring performance and accuracy"):
        input_features = extract_input_features(data_item)

        with time_measurement():
            predicted_ids = ov_model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        ground_truths.append(data_item["text"])
        predictions.append(transcription[0])

    word_accuracy = (1 - wer(ground_truths, predictions, reference_transform=wer_standardize,
                             hypothesis_transform=wer_standardize)) * 100
    mean_whole_infer_time = sum(whole_infer_times)
    mean_encoder_infer_time = sum(encoder_infer_times)
    mean_decoder_with_time_infer_time = sum(decoder_with_past_infer_times)
    return word_accuracy, (mean_whole_infer_time, mean_encoder_infer_time, mean_decoder_with_time_infer_time)

test_dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True)
test_dataset = test_dataset.shuffle(seed=42).take(TEST_DATASET_SIZE)
test_samples = [sample for sample in test_dataset]

accuracy_original, times_original = calculate_transcription_time_and_accuracy(ov_distil_model, test_samples)
accuracy_quantized, times_quantized = calculate_transcription_time_and_accuracy(ov_quantized_distil_model, test_samples)
print(f"Encoder performance speedup: {times_original[1] / times_quantized[1]:.3f}")
print(f"Decoder with past performance speedup: {times_original[2] / times_quantized[2]:.3f}")
print(f"Whole pipeline performance speedup: {times_original[0] / times_quantized[0]:.3f}")
print(f"Whisper transcription word accuracy. Original model: {accuracy_original:.2f}%. Quantized model: {accuracy_quantized:.2f}%.")
print(f"Accuracy drop: {accuracy_original - accuracy_quantized:.2f}%.")









