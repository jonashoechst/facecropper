# -*- coding: utf-8 -*-

import argparse
import logging
import os
import sys

import numpy as np
from cv2 import cv2


console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.propagate = False
logger.addHandler(console)


def cropfill(img: np.ndarray,
             top: int,
             right: int,
             bottom: int,
             left: int,
             ):

    dimen = (top,
             len(img[1]) - right,
             len(img) - bottom,
             left,
             )

    for i, size in enumerate(dimen):
        logger.debug(f"cropfill {i}: {size}")

        if size < 0:
            img = np.insert(img, 0, [img[0]] * -size, 0)
        else:
            img = img[size:]
        img = np.rot90(img, k=1, axes=(0, 1))

    return img


def extract_faces(img: np.ndarray,
                  cascade: cv2.CascadeClassifier,
                  spacing: float = 0.0,
                  force_square: bool = True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    faces = []

    for (x, y, width, height) in face_boxes:
        # force width equal to height
        if force_square:
            width = height = max(width, height)
        logger.info(f"detected face of size {width}x{height}")

        # compute additional absolute spacing
        space_horizontal = int(spacing * width)
        space_vertical = int(spacing * height)
        logger.debug(f"face with spacing " +
                     f"{width + space_horizontal*2}x" +
                     f"{height + space_vertical*2}")

        left = x - space_horizontal
        top = y - space_vertical
        right = x + width + space_horizontal
        bottom = y + height + space_vertical

        # compute filling points
        face = cropfill(img, top, right, bottom, left)

        faces.append(face)

    return faces


def circle_mask(img: np.ndarray,
                color=(255, 255, 255, 255)):
    if img.shape[0] != img.shape[1]:
        raise Exception(
            f"Image is non-square ({img.shape[0]}x{img.shape[1]}), " +
            "cannot apply circle mask.")

    # convert to 4-channel image (including alpha)
    if img.shape[2] < 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    mask = np.zeros(img.shape, img.dtype)

    cv2.circle(
        img=mask,
        center=(int(mask.shape[0]/2),
                int(mask.shape[1]/2)),
        radius=int(mask.shape[0]/2),
        color=color,
        # thickness -1: fill inner circle
        thickness=-1,
    )

    masked = cv2.bitwise_and(img, mask)
    return masked


def export(img: np.ndarray,
           output_path: str,
           size: int = 0,
           grayscale: bool = False):
    if size > 0:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    if grayscale:
        img_gray = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        _, _, _, alpha = cv2.split(img)
        luminescence = img_gray[0]
        img = cv2.merge((luminescence, luminescence, luminescence, alpha))

    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    cv2.imwrite(output_path, img)


def process_image(image_path: str,
                  output_template: str,
                  cascade: cv2.CascadeClassifier,
                  spacing: float,
                  size: int,
                  grayscale: bool):
    img = cv2.imread(image_path)
    logger.info(
        f"Processing {image_path}, resolution: {len(img)}x{len(img[0])}")

    # extract variables for output filename generation
    path = os.path.dirname(image_path)
    name, ext = os.path.splitext(os.path.basename(image_path))
    ext = ext[1:]

    # extract faces by their matched bounding boxes
    faces = extract_faces(img, cascade, spacing=spacing)
    logging.info(f"Found {len(faces)} faces")

    for i, face in enumerate(faces):
        masked = circle_mask(face)
        output_path = output_template.format(**locals())
        logger.info(f"Exporting {output_path}")
        export(masked, output_path, size, grayscale)


def main():
    class NewlineFormatter(argparse.HelpFormatter):
        def _split_lines(self, text, width):
            return text.splitlines()

    parser = argparse.ArgumentParser(
        description="Detect and crop faces from an image.",
        formatter_class=NewlineFormatter,
    )
    parser.add_argument(
        "image",
        nargs="+",
        help="path to input image",
    )
    parser.add_argument(
        "--cascade",
        default="haarcascade_frontalface_default.xml",
        help="face detection cascade to be used by OpenCV",
    )
    parser.add_argument(
        "-o", "--output",
        default="{name}_{i}.png",
        help="Output path template, evaluates placehoders: \n\
{path} -> original file path, \n\
{name} -> original file name, \n\
{ext} -> original file extension, \n\
{i} -> index of detected face",
    )
    parser.add_argument(
        "-p", "--padding",
        type=float,
        default="0.3",
        help="relative space around recognized face (> 0)",
    )
    parser.add_argument(
        "-s", "--size",
        type=int,
        default="200",
        help="maximum image resolution",
    )
    parser.add_argument(
        "-g", "--grayscale",
        type=bool,
        default=True,
        help="grayscale cropped image",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase verbosity (may be applied multiple times)",
        action="count",
        default=0
    )
    args = parser.parse_args()

    logging_level = max(10, logging.WARN - (args.verbose) * 10)
    logger.setLevel(logging_level)
    logger.debug(f"set logging level to {logging_level}")

    cascade_module = os.path.join(os.path.dirname(
        __file__), f"haarcascades/{args.cascade}")

    if os.path.exists(args.cascade):
        logger.info(f"Loading {args.cascade}")
        cascade = cv2.CascadeClassifier(args.cascade)
    elif os.path.exists(cascade_module):
        logger.info(f"Loading {cascade_module}")
        cascade = cv2.CascadeClassifier(cascade_module)
    else:
        logger.fatal(f"cascade could not be loaded, path: {cascade_module}")
        sys.exit(1)

    for image_path in args.image:
        try:
            process_image(image_path=image_path,
                          output_template=args.output,
                          cascade=cascade,
                          spacing=args.padding,
                          size=args.size,
                          grayscale=args.grayscale,
                          )
        except Exception as e:
            logger.error(f"{image_path} could not be processed: {e}")
