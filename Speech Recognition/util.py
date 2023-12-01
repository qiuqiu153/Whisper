from pathlib import Path
from tqdm import tqdm
from itertools import islice
from typing import List, Any
from openvino import Tensor
import shutil
import nncf
import gc
import openvino as ov
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from datasets import load_dataset
from init import*

def extract_input_features(sample):
    input_features = processor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
    ).input_features
    return input_features

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
def quantize(ov_model, calibration_dataset_size=10):
    
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
    quantized_ov_model.to(ov_model.device)
    quantized_ov_model.compile()
    return quantized_ov_model

