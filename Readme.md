facecropper
=== 

`facecropper` is tool to extract faces from images. It facilitates OpenCV to detect faces, grayscale and create circular images.

There are multiple settings, such as the padding, background color, or grayscaling:

![lenna.png processed by facecropper with default settings](docs/lenna_0.png)
![export using the grayscale option](docs/lenna_1.png)
![applying custom padding](docs/lenna_2.png)
![custom background color](docs/lenna_3.png)

```bash
facecropper -v -o docs/lenna_0.png docs/lenna.png
facecropper -v -o docs/lenna_1.png docs/lenna.png --grayscale
facecropper -v -o docs/lenna_2.png docs/lenna.png --grayscale --padding 1.0
facecropper -v -o docs/lenna_3.png docs/lenna.png --padding 1.0 --color "(122, 90, 50, 255)"
```

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
                   [-g] [-v] [-c COLOR]
                   image [image ...]

Detect and crop faces from an image.

positional arguments:
  image                 path to input image

optional arguments:
  -h, --help            show this help message and exit
  --cascade CASCADE     face detection cascade to be used by OpenCV
  -o OUTPUT, --output OUTPUT
                        Output path template, evaluates placehoders: 
                        {path} -> original file path, 
                        {name} -> original file name, 
                        {ext} -> original file extension, 
                        {i} -> index of detected face
  -p PADDING, --padding PADDING
                        relative space around recognized face (> 0), default=0.3
  -s SIZE, --size SIZE  export image resolution height / width, default=200
  -g, --grayscale       grayscale cropped image
  -v, --verbose         increase verbosity (may be applied multiple times)
  -c COLOR, --color COLOR
                        background color for circular cutout, BRG(A)-format, default: (255, 255, 255, 0)
```

### Cascades

The cascades used by OpenCV can be 

- user-supplied by giving a path to the respective xml file or 
- be selected from the project:

```
haarcascade_eye.xml
haarcascade_eye_tree_eyeglasses.xml
haarcascade_frontalcatface.xml
haarcascade_frontalcatface_extended.xml
haarcascade_frontalface_alt.xml
haarcascade_frontalface_alt2.xml
haarcascade_frontalface_alt_tree.xml
haarcascade_frontalface_default.xml
haarcascade_fullbody.xml
haarcascade_lefteye_2splits.xml
haarcascade_licence_plate_rus_16stages.xml
haarcascade_lowerbody.xml
haarcascade_profileface.xml
haarcascade_righteye_2splits.xml
haarcascade_russian_plate_number.xml
haarcascade_smile.xml
haarcascade_upperbody.xml
```

## Example

```bash
$ facecropper -v -o docs/lenna_3.png docs/lenna.png --padding 1.0 --color "(122, 90, 50, 255)"
INFO: Loading /Users/hoechst/Projects/facecropper/facecropper/haarcascades/haarcascade_frontalface_default.xml
INFO: Processing docs/lenna.png, resolution: 512x512
INFO: detected face of size 173x173
INFO: Exporting docs/lenna_3.png, grayscale: False
```
