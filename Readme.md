facecropper
=== 

`facecropper?` is tool to extract faces from images.

## Installation

`facecropper` comes a python package and thus can be installed using pip.

```bash
git clone https://github.com/jonashoechst/facecropper.git
pushd facecropper
pip install .
```

## Usage

```bash
$ facecropper -h
usage: facecropper [-h] [--cascade CASCADE] [-o OUTPUT] [-p PADDING] [-s SIZE]
                   [-g GRAYSCALE]
                   image [image ...]

Detect and crop faces from an image.

positional arguments:
  image                 path to input image

optional arguments:
  -h, --help            show this help message and exit
  --cascade CASCADE     face detection cascade to be used by OpenCV
  -o OUTPUT, --output OUTPUT
                        Output path template, evaluates placehoders: {path} ->
                        original file path, {name} -> original file name,
                        {ext} -> original file extension, {i} -> index of
                        detected face
  -p PADDING, --padding PADDING
                        relative space around recognized face (> 0)
  -s SIZE, --size SIZE  maximum image resolution
  -g GRAYSCALE, --grayscale GRAYSCALE
                        grayscale cropped image
```

## Example

```
$ facecropper lenna.png 
Loading haarcascades/haarcascade_frontalface_default.xml
Processing lenna.png
Exporting lenna_0.png
INFO:facecropper:Exporting lenna_0.png
Done.
INFO:facecropper:Done.
```
