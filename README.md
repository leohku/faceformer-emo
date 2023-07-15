# FaceFormer Emo

## Abstract

> We propose FaceFormer Emo, a Transformer-based speech to 3D face mesh model that produces highly emotional expressions from the input audio.
>
> This work extends FaceFormer in producing realistic and expressive outputs without a large loss in lip vertex accuracy, and introduces a novel lip vertex loss function which increases the weights of the vertices nearby the lips to increase accuracy and emotion dynamics.

## Poster

<p align="center">
<img src="https://raw.githubusercontent.com/leohku/faceformer-emo/main/poster.jpeg" width="70%" />
</p>

## 3-min Video Introduction

[[YouTube Link]](https://www.youtube.com/watch?v=LO83GkiqRXQ)

> [![FaceFormer Emo Introduction Video](https://img.youtube.com/vi/LO83GkiqRXQ/0.jpg)](https://www.youtube.com/watch?v=LO83GkiqRXQ)

## FaceFormer

PyTorch implementation for the paper:

> **FaceFormer: Speech-Driven 3D Facial Animation with Transformers**, ***CVPR 2022***.
>
> Yingruo Fan, Zhaojiang Lin, Jun Saito, Wenping Wang, Taku Komura
>
> [[Paper]](https://arxiv.org/pdf/2112.05329.pdf) [[Project Page]](https://evelynfan.github.io/audio2face/) 

<p align="center">
<img src="framework.jpg" width="70%" />
</p>

> Given the raw audio input and a neutral 3D face mesh, our proposed end-to-end Transformer-based architecture, FaceFormer, can autoregressively synthesize a sequence of realistic 3D facial motions with accurate lip movements.

### Environment

- Ubuntu 18.04.1
- Python 3.7
- Pytorch 1.9.0

### Dependencies

- Check the required python packages in `requirements.txt`.
- ffmpeg
- [MPI-IS/mesh](https://github.com/MPI-IS/mesh)

### Data

#### VOCASET

Request the VOCASET data from [https://voca.is.tue.mpg.de/](https://voca.is.tue.mpg.de/). Place the downloaded files `data_verts.npy`, `raw_audio_fixed.pkl`, `templates.pkl` and `subj_seq_to_idx.pkl` in the folder `VOCASET`. Download "FLAME_sample.ply" from [voca](https://github.com/TimoBolkart/voca/tree/master/template) and put it in `VOCASET/templates`.

#### BIWI

Request the BIWI dataset from [Biwi 3D Audiovisual Corpus of Affective Communication](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html). The dataset contains the following subfolders:

- 'faces' contains the binary (.vl) files for the tracked facial geometries. 
- 'rigid_scans' contains the templates stored as .obj files. 
- 'audio' contains audio signals stored as .wav files. 

Place the folders 'faces' and 'rigid_scans' in `BIWI` and place the wav files in `BIWI/wav`.

### Demo

Download the pretrained models from [biwi.pth](https://drive.google.com/file/d/1WR1P25EE7Aj1nDZ4MeRsqdyGnGzmkbPX/view?usp=sharing) and [vocaset.pth](https://drive.google.com/file/d/1GUQBk9FqUimoT6UNgU0gyQnjGv-2_Lyp/view?usp=sharing). Put the pretrained models under `BIWI` and `VOCASET` folders, respectively. Given the audio signal,

- to animate a mesh in BIWI topology, run: 
	```
	python demo.py --model_name biwi --wav_path "demo/wav/test.wav" --dataset BIWI --vertice_dim 70110  --feature_dim 128 --period 25 --fps 25 --train_subjects "F2 F3 F4 M3 M4 M5" --test_subjects "F1 F5 F6 F7 F8 M1 M2 M6" --condition M3 --subject M1
	```

- to animate a mesh in FLAME topology, run: 
	```
	python demo.py --model_name vocaset --wav_path "demo/wav/test.wav" --dataset vocaset --vertice_dim 15069 --feature_dim 64 --period 30  --fps 30  --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA" --test_subjects "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA" --condition FaceTalk_170913_03279_TA --subject FaceTalk_170809_00138_TA
	```
	This script will automatically generate the rendered videos in the `demo/output` folder. You can also put your own test audio file (.wav format) under the `demo/wav` folder and specify the argument `--wav_path "demo/wav/test.wav"` accordingly.

### Training and Testing on VOCASET

####  Data Preparation

- Read the vertices/audio data and convert them to .npy/.wav files stored in `vocaset/vertices_npy` and `vocaset/wav`:

	```
	cd VOCASET
	python process_voca_data.py
	```

#### Training and Testing

(WARNING: The snippet below might exhaust all memory on GPU, use 5 train_subjects for safe measure)

- To train the model on VOCASET and obtain the results on the testing set, run:

	```
	python main.py --dataset vocaset --vertice_dim 15069 --feature_dim 64 --period 30 --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA" --val_subjects "FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA" --test_subjects "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA"
	```
	The results and the trained models will be saved to `vocaset/result` and `vocaset/save`.


#### Visualization

- To visualize the results, run:

	```
	python render.py --dataset vocaset --vertice_dim 15069 --fps 30
	```
	You can find the outputs in the `vocaset/output` folder.

### Acknowledgement

We gratefully acknowledge ETHZ-CVL for providing the [B3D(AC)2](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html) database and MPI-IS for releasing the [VOCASET](https://voca.is.tue.mpg.de/) dataset. The implementation of wav2vec2 is built upon [huggingface-transformers](https://github.com/huggingface/transformers/blob/master/src/transformers/models/wav2vec2/modeling_wav2vec2.py), and the temporal bias is modified from [ALiBi](https://github.com/ofirpress/attention_with_linear_biases). We use [MPI-IS/mesh](https://github.com/MPI-IS/mesh) for mesh processing and [VOCA/rendering](https://github.com/TimoBolkart/voca) for rendering. We thank the authors for their excellent works. Any third-party packages are owned by their respective authors and must be used under their respective licenses.


