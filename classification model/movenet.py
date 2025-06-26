# movenet.py
import os
from typing import Dict, List

import cv2
import numpy as np
import tensorflow as tf  # Import TensorFlow directly for TFLite Interpreter
from data import BodyPart, person_from_keypoints_with_scores, Person

# Alias for easier use
Interpreter = tf.lite.Interpreter


class Movenet:
    """A wrapper class for a MoveNet TensorFlow Lite pose estimation model."""

    # Threshold for considering keypoints valid for cropping
    _MIN_CROP_KEYPOINT_SCORE = 0.2
    # Expansion ratios used for computing the crop region based on torso/body
    _TORSO_EXPANSION_RATIO = 1.9
    _BODY_EXPANSION_RATIO = 1.2

    def __init__(self, model_name: str) -> None:
        """
        Initialize the MoveNet interpreter and retrieve model input/output details.
        """
        # Append file extension if not provided
        _, ext = os.path.splitext(model_name)
        if not ext:
            model_name += ".tflite"

        # Load the TFLite model with 4 threads for performance
        self._interpreter = Interpreter(model_path=model_name, num_threads=4)
        self._interpreter.allocate_tensors()

        # Get input/output tensor details
        inp = self._interpreter.get_input_details()[0]
        out = self._interpreter.get_output_details()[0]

        self._input_index = inp["index"]
        self._output_index = out["index"]
        self._input_height = inp["shape"][1]
        self._input_width = inp["shape"][2]
        self._input_dtype = inp["dtype"]

        # Stores the region of interest to crop in the next frame
        self._crop_region = None

    def detect(self, input_image: np.ndarray, reset_crop_region=False) -> Person:
        """
        Runs pose detection on the input image and returns a Person object.

        Args:
            input_image: RGB image as a NumPy array.
            reset_crop_region: Whether to reinitialize the crop region.

        Returns:
            Person: The detected pose.
        """
        h, w, _ = input_image.shape

        # Initialize or reset the crop region based on the whole image
        if self._crop_region is None or reset_crop_region:
            self._crop_region = self.init_crop_region(h, w)

        # Crop and resize the image to fit model input size
        cropped = self._crop_and_resize(input_image, self._crop_region)

        # Run inference
        self._interpreter.set_tensor(self._input_index, cropped)
        self._interpreter.invoke()
        kps = self._interpreter.get_tensor(self._output_index)
        kps_np = np.squeeze(kps)

        # Update the crop region for next frame
        self._crop_region = self._determine_crop_region(kps_np, h, w)

        # Return structured pose as Person
        return person_from_keypoints_with_scores(kps_np, h, w)

    def _crop_and_resize(self, image, crop_region) -> np.ndarray:
        """
        Crop and resize the image based on normalized crop coordinates.

        Returns:
            Preprocessed image tensor ready for model input.
        """
        # Convert normalized crop coordinates to absolute pixels
        y0 = int(max(0, crop_region["y_min"] * image.shape[0]))
        x0 = int(max(0, crop_region["x_min"] * image.shape[1]))
        y1 = int(min(image.shape[0], crop_region["y_max"] * image.shape[0]))
        x1 = int(min(image.shape[1], crop_region["x_max"] * image.shape[1]))

        cropped = image[y0:y1, x0:x1, :]
        resized = cv2.resize(cropped, (self._input_width, self._input_height))

        # Add batch dimension
        tensor = np.expand_dims(resized, axis=0)

        # Normalize or cast to model-required dtype
        if self._input_dtype == np.float32:
            tensor = (tensor.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
        elif self._input_dtype == np.uint8:
            tensor = tensor.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported model input type: {self._input_dtype}")

        return tensor

    def init_crop_region(self, h: int, w: int) -> Dict[str, float]:
        """
        Initializes the crop region to the center square of the image.

        Ensures that the initial region maintains aspect ratio suitable for the model.
        """
        if w > h:
            # Wider image: pad height
            y_min = (h / 2 - w / 2) / h
            return {
                "y_min": y_min,
                "x_min": 0.0,
                "y_max": y_min + w / h,
                "x_max": 1.0,
                "height": w / h,
                "width": 1.0,
            }
        else:
            # Taller image: pad width
            x_min = (w / 2 - h / 2) / w
            return {
                "y_min": 0.0,
                "x_min": x_min,
                "y_max": 1.0,
                "x_max": x_min + h / w,
                "height": 1.0,
                "width": h / w,
            }

    def _torso_visible(self, kps: np.ndarray) -> bool:
        """
        Checks if torso keypoints are visible above a threshold.

        Used to decide whether we can trust the pose enough to recalculate the crop.
        """
        scores = {
            part: kps[part.value, 2] for part in (
                BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP,
                BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER
            )
        }
        return (
            (scores[BodyPart.LEFT_HIP] > Movenet._MIN_CROP_KEYPOINT_SCORE or
             scores[BodyPart.RIGHT_HIP] > Movenet._MIN_CROP_KEYPOINT_SCORE)
            and
            (scores[BodyPart.LEFT_SHOULDER] > Movenet._MIN_CROP_KEYPOINT_SCORE or
             scores[BodyPart.RIGHT_SHOULDER] > Movenet._MIN_CROP_KEYPOINT_SCORE)
        )

    def _determine_torso_and_body_range(self, kps, tgt, cy, cx):
        """
        Calculates the maximum distance of torso/body keypoints from the center point.

        Used to determine how large the next crop should be.
        """
        # Check torso joints for expansion range
        torso_joints = [BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER,
                        BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP]
        max_torso_yr, max_torso_xr = 0.0, 0.0
        for j in torso_joints:
            dy = abs(cy - tgt[j][0])
            dx = abs(cx - tgt[j][1])
            max_torso_yr = max(max_torso_yr, dy)
            max_torso_xr = max(max_torso_xr, dx)

        # Check all visible keypoints for full-body expansion
        max_body_yr, max_body_xr = 0.0, 0.0
        for idx in range(len(BodyPart)):
            if kps[idx, 2] < Movenet._MIN_CROP_KEYPOINT_SCORE:
                continue
            dy = abs(cy - tgt[BodyPart(idx)][0])
            dx = abs(cx - tgt[BodyPart(idx)][1])
            max_body_yr = max(max_body_yr, dy)
            max_body_xr = max(max_body_xr, dx)

        return max_torso_yr, max_torso_xr, max_body_yr, max_body_xr

    def _determine_crop_region(self, kps, h, w):
        """
        Determines the next crop region based on current keypoints.

        Uses torso and body joint visibility and spacing to recenter the crop.
        Falls back to default if torso isn't confidently detected.
        """
        # Convert keypoint coordinates to pixel values
        tgt = {
            BodyPart(idx): [kps[idx, 0] * h, kps[idx, 1] * w]
            for idx in range(len(BodyPart))
        }

        if self._torso_visible(kps):
            # Calculate center between hips
            cy = (tgt[BodyPart.LEFT_HIP][0] + tgt[BodyPart.RIGHT_HIP][0]) / 2
            cx = (tgt[BodyPart.LEFT_HIP][1] + tgt[BodyPart.RIGHT_HIP][1]) / 2

            # Get expansion range
            ytr, xtr, ybr, xbr = self._determine_torso_and_body_range(kps, tgt, cy, cx)

            # Determine half-size of square crop (scaled by expansion ratios)
            half = np.amin([
                np.amax([xtr * Movenet._TORSO_EXPANSION_RATIO,
                         ytr * Movenet._TORSO_EXPANSION_RATIO,
                         xbr * Movenet._BODY_EXPANSION_RATIO,
                         ybr * Movenet._BODY_EXPANSION_RATIO]),
                np.amax([cx, w - cx, cy, h - cy])  # constrain to image bounds
            ])

            # Fallback if the region is too large (probably bad detection)
            if half > max(h, w) / 2:
                return self.init_crop_region(h, w)

            # Convert back to normalized crop coordinates
            y0 = (cy - half) / h
            x0 = (cx - half) / w
            length = (half * 2) / h  # height in normalized units
            return {
                "y_min": y0,
                "x_min": x0,
                "y_max": y0 + length,
                "x_max": x0 + length * (h / w),
                "height": length,
                "width": length * (h / w),
            }
        else:
            # If torso is not visible, use default crop
            return self.init_crop_region(h, w)
