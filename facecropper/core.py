# -*- coding: utf-8 -*-

import argparse
import logging
import os

import numpy as np
from cv2 import cv2

logger = logging.getLogger("facecropper")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


def extract_faces(img, cascade, spacing=0.0, force_square=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_boxes = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    faces = []

    for (x, y, width, height) in face_boxes:
        # force width equal to height
        if force_square:
            width = height = max(width, height)

        # compute additional absolute spacing
        space_horizontal = int(spacing * width)
        space_vertical = int(spacing * height)

        # crop out face, including the calculated space
        face = img[
            y-space_vertical: y+height+space_vertical,
            x-space_horizontal: x+width+space_horizontal,
        ]

        faces.append(face)

    return faces


def circle_mask(img, color=(255, 255, 255, 255)):
    if img.shape[0] != img.shape[1]:
        raise Exception(
            f"Image is non-square ({img.shape[0]}x{img.shape[1]}), \
                cannot apply circle mask.")

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


def export(img, output_path: str, size: int = 0, grayscale: bool = False):
    if size > 0:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    if grayscale:
        img_gray = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        _, _, _, alpha = cv2.split(img)
        luminescence = img_gray[0]
        img = cv2.merge((luminescence, luminescence, luminescence, alpha))

    cv2.imwrite(output_path, img)


def process_image(image_path, output_template, cascade, spaceing, size,
                  grayscale):
    logger.info(f"Processing {image_path}")
    img = cv2.imread(image_path)
    path = os.path.dirname(image_path)
    name, ext = os.path.splitext(os.path.basename(image_path))
    ext = ext[1:]

    faces = extract_faces(img, cascade, spacing=spaceing)
    logging.info(f"Found {len(faces)} faces")

    for i, face in enumerate(faces):
        masked = circle_mask(face)
        output_path = output_template.format(**locals())
        logger.info(f"Exporting {output_path}")
        export(masked, output_path, size, grayscale)


def main():
    parser = argparse.ArgumentParser(
        description="Detect and crop faces from an image.",
    )
    parser.add_argument(
        "image",
        nargs="+",
        help="path to input image",
    )
    parser.add_argument(
        "--cascade",
        default="haarcascades/haarcascade_frontalface_default.xml",
        help="face detection cascade to be used by OpenCV",
    )
    parser.add_argument(
        "-o", "--output",
        default="{name}_{i}.png",
        help="Output path template, evaluates placehoders: \
            {path} -> original file path, \
            {name} -> original file name, \
            {ext} -> original file extension, \
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

    args = parser.parse_args()

    logger.info(f"Loading {args.cascade}")
    cascade = cv2.CascadeClassifier(args.cascade)

    for image_path in args.image:
        try:
            process_image(image_path=image_path,
                          output_template=args.output,
                          cascade=cascade,
                          spaceing=args.padding,
                          size=args.size,
                          grayscale=args.grayscale,
                          )
        except Exception as e:
            logger.warning(f"Error processing {image_path}: {e}")

    logger.info("Done.")
