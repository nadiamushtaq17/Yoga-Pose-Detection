import tensorflow as tf
import numpy as np
import pandas as pd
import os
from movenet import Movenet  # Assumes movenet.py wrapper is implemented
import wget
import csv
import tqdm
from data import BodyPart

# ---- Download MoveNet Thunder model if not already present ----
if "movenet_thunder.tflite" not in os.listdir():
    wget.download(
        "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite",
        "movenet_thunder.tflite",
    )

# Initialize MoveNet wrapper
movenet = Movenet("movenet_thunder")

# ---- Pose detection helper ----
def detect(input_tensor, inference_count=3):
    """
    Runs MoveNet detection on the input image.
    Applies padding and resizes to 256x256.
    Runs multiple inferences to refine crop region.
    """
    input_tensor = tf.image.resize_with_pad(input_tensor, 256, 256)
    input_tensor = tf.cast(input_tensor, tf.uint8)
    
    # First detection with reset crop
    movenet.detect(input_tensor.numpy(), reset_crop_region=True)
    
    # Run additional inferences for refinement
    for _ in range(inference_count - 1):
        detection = movenet.detect(input_tensor.numpy(), reset_crop_region=False)
    
    return detection


# ---- Preprocessing pipeline class ----
class Preprocessor:
    def __init__(self, images_in_folder, csvs_out_path):
        """
        Args:
            images_in_folder (str): Folder containing pose-class subfolders of images.
            csvs_out_path (str): Final CSV file path for merged output.
        """
        self._images_in_folder = images_in_folder
        self._csvs_out_path = csvs_out_path
        self._csvs_out_folder_per_class = "csv_per_pose"
        self._message = []

        if not os.path.isdir(self._csvs_out_folder_per_class):
            os.makedirs(self._csvs_out_folder_per_class)

        # Each subfolder is assumed to be a pose class (e.g., 'tree', 'warrior')
        self._pose_class_names = sorted(os.listdir(images_in_folder))

    def process(self, detection_threshold=0.1):
        """
        Processes all images, runs pose detection, filters based on score,
        and saves coordinates to class-specific CSVs.
        """
        for cls in self._pose_class_names:
            in_dir = os.path.join(self._images_in_folder, cls)
            out_csv = os.path.join(self._csvs_out_folder_per_class, f"{cls}.csv")

            with open(out_csv, "w", newline="") as f:
                writer = csv.writer(f)

                for img_name in tqdm.tqdm(sorted(os.listdir(in_dir))):
                    p = os.path.join(in_dir, img_name)

                    # Load image safely
                    try:
                        img = tf.io.decode_jpeg(tf.io.read_file(p))
                    except:
                        self._message.append(f"Skipped {p} (bad file)")
                        continue

                    if img.shape[-1] != 3:
                        self._message.append(f"Skipped {p} (not RGB)")
                        continue

                    # Run pose detection
                    try:
                        person = detect(img)
                    except Exception as e:
                        self._message.append(f"Failed on {p}: {e}")
                        continue

                    # Check for low-confidence keypoints
                    scores = [kp.score for kp in person.keypoints]
                    if min(scores) < detection_threshold:
                        self._message.append(f"Skipped {p} (low score)")
                        continue

                    # Flatten keypoint coordinates to CSV row
                    coords = np.array(
                        [[kp.coordinate.x, kp.coordinate.y, kp.score] for kp in person.keypoints],
                        dtype=np.float32
                    ).flatten().astype(str).tolist()

                    writer.writerow([img_name] + coords)

        # Show skipped/missing warnings
        print("\n".join(self._message))

        # Merge all per-class CSVs into one labeled DataFrame
        df = self.all_landmarks_as_dataframe()
        df.to_csv(self._csvs_out_path, index=False)

    def all_landmarks_as_dataframe(self):
        """
        Combines individual pose-class CSVs into a single dataframe.
        Adds class labels and standard column names.
        """
        dfs = []
        for idx, cls in enumerate(self._pose_class_names):
            path = os.path.join(self._csvs_out_folder_per_class, f"{cls}.csv")
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                continue
            try:
                sub = pd.read_csv(path, header=None)
            except pd.errors.EmptyDataError:
                continue

            sub["class_no"] = idx
            sub["class_name"] = cls
            sub.iloc[:, 0] = cls + "/" + sub.iloc[:, 0].astype(str)  # prefix filename with class
            dfs.append(sub)

        if not dfs:
            print("No valid CSV data found.")
            return pd.DataFrame()

        df = pd.concat(dfs, ignore_index=True)

        # Build CSV header: filename, x/y/score for each keypoint, then labels
        coord_cols = [
            f"{bp.name}_{axis}" for bp in BodyPart for axis in ("x", "y", "score")
        ]
        header = ["filename"] + coord_cols + ["class_no", "class_name"]
        df.columns = header
        return df


# ---- Main execution ----
if __name__ == "__main__":
    # Process training images
    prep = Preprocessor("yoga_poses/train", "train_data.csv")
    prep.process()

    # Process test images
    prep = Preprocessor("yoga_poses/test", "test_data.csv")
    prep.process()
