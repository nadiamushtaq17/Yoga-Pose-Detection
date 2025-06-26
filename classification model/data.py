"""Module contains the data types used in pose estimation.

This module defines the core data structures to represent body parts, keypoints,
bounding boxes, detected persons, and classification categories. It also provides
a utility function to convert raw keypoint data into a structured Person object.
"""

import enum
from typing import List, NamedTuple

import numpy as np


class BodyPart(enum.Enum):
    """Enumeration of human body keypoints detected by pose estimation models.

    These are standard anatomical landmarks typically predicted by pose estimation
    models such as MoveNet or BlazePose. The indices correspond to a consistent
    order expected in model output.
    """
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class Point(NamedTuple):
    """A point in 2D space, representing pixel coordinates (x, y)."""
    x: float
    y: float


class Rectangle(NamedTuple):
    """A rectangle in 2D space, defined by two corner points."""
    start_point: Point  # Top-left or minimum coordinate corner
    end_point: Point    # Bottom-right or maximum coordinate corner


class KeyPoint(NamedTuple):
    """A single detected keypoint of the human body.

    Attributes:
        body_part: The specific body part this keypoint represents.
        coordinate: The (x, y) location of the keypoint in image space.
        score: Confidence score from the pose estimation model.
    """
    body_part: BodyPart
    coordinate: Point
    score: float


class Person(NamedTuple):
    """Represents a full-body pose detected by the pose estimation model.

    Attributes:
        keypoints: List of keypoints that make up the pose.
        bounding_box: A rectangle tightly enclosing all valid keypoints.
        score: Average confidence score for selected keypoints.
        id: Optional identifier for tracking individuals across frames.
    """
    keypoints: List[KeyPoint]
    bounding_box: Rectangle
    score: float
    id: int = None


def person_from_keypoints_with_scores(
    keypoints_with_scores: np.ndarray,
    image_height: float,
    image_width: float,
    keypoint_score_threshold: float = 0.1
) -> Person:
    """Creates a Person instance from a single pose estimation model output.

    Converts raw keypoint data (normalized coordinates and scores) into structured
    KeyPoint and Person objects, including bounding box computation.

    Args:
        keypoints_with_scores: Numpy array of shape [17, 3] containing
            [y, x, score] for each keypoint.
        image_height: Height of the image in pixels.
        image_width: Width of the image in pixels.
        keypoint_score_threshold: Only keypoints with a score above this
            threshold contribute to the overall person score.

    Returns:
        A Person object containing structured keypoints and metadata.
    """
    kpts_x = keypoints_with_scores[:, 1]
    kpts_y = keypoints_with_scores[:, 0]
    scores = keypoints_with_scores[:, 2]

    # Convert keypoints to image space and wrap in KeyPoint structure
    keypoints = [
        KeyPoint(
            body_part=BodyPart(i),
            coordinate=Point(int(kpts_x[i] * image_width), int(kpts_y[i] * image_height)),
            score=scores[i]
        ) for i in range(scores.shape[0])
    ]

    # Compute bounding box from keypoint extents
    start_point = Point(
        int(np.amin(kpts_x) * image_width),
        int(np.amin(kpts_y) * image_height)
    )
    end_point = Point(
        int(np.amax(kpts_x) * image_width),
        int(np.amax(kpts_y) * image_height)
    )
    bounding_box = Rectangle(start_point, end_point)

    # Compute average score from keypoints above threshold
    scores_above_threshold = list(filter(lambda x: x > keypoint_score_threshold, scores))
    person_score = np.average(scores_above_threshold) if scores_above_threshold else 0.0

    return Person(keypoints, bounding_box, person_score)


class Category(NamedTuple):
    """A classification category with label and confidence score.

    Useful for secondary classification tasks like gesture recognition,
    activity classification, or object detection labels.

    Attributes:
        label: The name of the category (e.g., "standing", "jumping").
        score: Model confidence score for this label.
    """
    label: str
    score: float
