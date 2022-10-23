import numpy as np
import torch
import torch.nn as nn
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model as W2V,
    Wav2Vec2PreTrainedModel,
)

class Wav2VecEmo():
    MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(self.MODEL_ID)
        
    def forward(self, audio):
        self.processor(audio)

class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x).to(device='cuda')
        x = self.dropout(x)
        x = self.out_proj(x)

        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = W2V(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1).to(device='cuda')
        logits = self.classifier(hidden_states)

        return hidden_states, logits

class Wav2VecEmoRobust():
    def __init__(self):
        self.model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = EmotionModel.from_pretrained(self.model_name).to(device='cuda')
    
    def process_func(
        self,
        x: np.ndarray,
        sampling_rate: int,
        embeddings: bool = False,
    ) -> np.ndarray:
        r"""Predict emotions or extract embeddings from raw audio signal."""

        # run through processor to normalize signal
        # always returns a batch, so we just get the first entry
        # then we put it on the device
        y = self.processor(x, sampling_rate=sampling_rate)
        y = y['input_values'][0]
        # y = torch.from_numpy(y)
        y = torch.from_numpy(y).float().to(device='cuda')
        print('y', y)
        # run through model
        with torch.no_grad():
            y = self.model(y)[0 if embeddings else 1]

        # convert to numpy
        y = y.detach().cpu().numpy()

        return y
    
if __name__ == '__main__':
    signal = torch.from_numpy(np.zeros((1, 16000), dtype=np.float32)).to(device='cuda')
    s2 = np.zeros((1, 16000), dtype=np.float32)
    print(s2)

    encoder = Wav2VecEmoRobust()
    y_0 = encoder.process_func(s2, 16000)
    print(y_0, len(y_0))
    y_1 = encoder.process_func(s2, 16000, True)
    print(y_1, len(y_1[0]))