# Installation

## Python
1. Had to install `wheel`
  - Else error: invalid command `bdist_wheel`
2. Remove `pickle` from requirements.txt
3. `numba` and `scipy` "incompatibilities" with numpy is okay, since numpy 1.23.1 is a maintanence release
4. Change `pyopengl` from 3.1.4 to 3.1.0 to avoid incompatibilities with `pyrender`

## ffmpeg
`sudo snap install ffmpeg`

## MPI-IS
`sudo apt-get install libboost-dev`
Boost includes are automatically at: `/usr/include/boost`

In project repo, `BOOST_INCLUDE_DIRS=/usr/include/boost make all`
`chmod 755 bin/meshviewer`

## Quick Start
In lambda environment, run

`source ~/venv/fenv/bin/activate`

## fix install bugs
```
demo.py
ImportError: ('Unable to load OpenGL library', 'OSMesa: cannot open shared object file: No such file or directory', 'OSMesa', None)

fix: sudo apt install -y python-opengl libosmesa6
```
```
demo.py
ModuleNotFoundError: No module named 'psbody'

fix: use the same venv as the project
```
```
demo.py
RuntimeError: The shape of the 3D attn_mask is torch.Size([4, 600, 600]), but should be (4, 601, 601).

when trimming the audio, use ffmpeg's command instead:
ffmpeg -ss 0 -i missile_3.wav -t 15 -c copy missile_4.wav

Somehow, this will cause librosa's load np.ndarray shape to match the shape of the 3D attn_mask in later steps, which is always off by one for some other reason.
```

---

# Demo

Download model biwi.pth
Download FLAME_sample.ply and place in vocaset/templates

# Peculiarities

1. Maximum input wav length for Biwi: 24 seconds (else, RuntimeError: The shape of the 3D attn_mask is torch.Size([4, 600, 600]), but should be (4, 601, 601).)
2. Maximum input wav length for VOCA: 20 seconds (again, same error)
3. The video rendered is missing one frame at the end (0.04s @ 25fps), compared to the audio.
4. The linear_interpolation of wav2vec outputs should be done with 49Hz -> 30Hz instead of 50Hz?
5. The "save memory" mechanism for vocaset, still needed?

# Training
# MEAD + EMOCA
Training command (25 epoch, 1 subject)

python main.py --dataset vocaset --vertice_dim 15354 --feature_dim 64 --period 30 --train_subjects "M005" --val_subjects "M005" --test_subjects "M005" --wav_path "/home/leoho/data/pipeline-data/pipeline-data-lambda/MEAD_FACEFORMER/wav" --vertices_path "/home/leoho/data/pipeline-data/pipeline-data-lambda/MEAD_FACEFORMER/vertices_npy" --template_file "/home/leoho/data/pipeline-data/pipeline-data-lambda/MEAD_FACEFORMER/template.pkl" --save_path "/home/leoho/data/pipeline-data/pipeline-data-lambda/MEAD_TRAINED/save" --result_path "/home/leoho/data/pipeline-data/pipeline-data-lambda/MEAD_TRAINED/result" --max_epoch 25

Needs ~10GB RAM

Training command (100 epoch, 5 subjects)

python main.py --dataset vocaset --vertice_dim 15354 --feature_dim 64 --period 30 --train_subjects "M005 W011 M009 W014" --val_subjects "M011" --test_subjects "M011" --wav_path "/home/leoho/data/pipeline-data/pipeline-data-lambda/MEAD_FACEFORMER/wav" --vertices_path "/home/leoho/data/pipeline-data/pipeline-data-lambda/MEAD_FACEFORMER/vertices_npy" --template_file "/home/leoho/data/pipeline-data/pipeline-data-lambda/MEAD_FACEFORMER/template.pkl" --save_path "/home/leoho/data/pipeline-data/pipeline-data-lambda/MEAD_TRAINED/save" --result_path "/home/leoho/data/pipeline-data/pipeline-data-lambda/MEAD_TRAINED/result" --variance_indices_path "/home/leoho/data/pipeline-data/pipeline-data-lambda/MEAD_FACEFORMER/variance_indices-600.pkl" --max_epoch 100

Needs ~30GB RAM

Demo command

python demo.py --model_name vocaset --dataset vocaset --vertice_dim 15354 --feature_dim 64 --period 30  --fps 30  --train_subjects "M005 W011 M009 W014" --test_subjects "M011" --template_path "/home/leoho/data/pipeline-data/pipeline-data-lambda/MEAD_FACEFORMER/template.pkl" --render_template_path "/home/leoho/data/pipeline-data/pipeline-data-lambda/MEAD_FACEFORMER/templates" --condition M005 --subject M009 --wav_path "demo/wav/test.wav" --variance_indices_path "/home/leoho/data/pipeline-data/pipeline-data-lambda/MEAD_FACEFORMER/variance_indices-600.pkl"

---

# VOCASET
Training command (25 epoch, 1 subject)

python main.py --dataset vocaset --vertice_dim 15069 --feature_dim 64 --period 30 --train_subjects "FaceTalk_170728_03272_TA" --val_subjects "FaceTalk_170728_03272_TA" --test_subjects "FaceTalk_170728_03272_TA" --max_epoch 25

Demo command

python demo.py --model_name vocaset --wav_path "demo/wav/test.wav" --dataset vocaset --vertice_dim 15069 --feature_dim 64 --period 30  --fps 30  --train_subjects "FaceTalk_170728_03272_TA" --test_subjects "FaceTalk_170728_03272_TA" --condition FaceTalk_170728_03272_TA --subject FaceTalk_170728_03272_TA

Render command
python render.py --dataset MEAD --subject_id M005 --vertice_dim 15354

# Original REPO VOCASET
Training command

python main.py --dataset vocaset --vertice_dim 15069 --feature_dim 64 --period 30 --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA" --val_subjects "FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA" --test_subjects "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA"

Demo command

python demo.py --model_name vocaset --wav_path "demo/wav/test.wav" --dataset vocaset --vertice_dim 15069 --feature_dim 64 --period 30  --fps 30  --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA" --test_subjects "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA" --condition FaceTalk_170913_03279_TA --subject FaceTalk_170809_00138_TA