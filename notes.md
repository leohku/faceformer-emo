# Installation

## Python
1. Had to install `wheel`
  - Else error: invalid command `bdist_wheel`
2. Remove `pickle` from requirements.txt
3. `numba` and `scipy` "incompatabilities" with numpy is okay, since numpy 1.23.1 is a maintanence release

## ffmpeg
`sudo snap install ffmpeg`

## MPI-IS
`sudo apt-get install libboost-dev`
Boost includes are automatically at: `/usr/include/boost`

In project repo, `BOOST_INCLUDE_DIRS=/usr/include/boost make all`
`chmod 755 bin/meshviewer`

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
2. The video rendered is missing one frame at the end (0.04s @ 25fps), compared to the audio.

