# pose_classifier.py

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from data import BodyPart

# ---- File paths ----
TRAIN_CSV = 'train_data.csv'
TEST_CSV  = 'test_data.csv'
MODEL_H5  = 'final_model.h5'


# ---- Data loading and preprocessing ----

def load_csv(csv_path):
    """
    Loads landmark CSV, extracts features (X) and labels (y), and one-hot encodes the labels.
    Assumes CSV has 'filename', 'class_name', and 'class_no' columns, and landmark values as features.
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['filename'])           # Drop filename, not useful for model
    classes = df.pop('class_name').unique()      # Store class labels (not used for training directly)
    y = df.pop('class_no')                       # Numeric class ID
    X = df.astype('float32')                     # Convert coordinates to float
    y = keras.utils.to_categorical(y, num_classes=len(classes))  # One-hot encode targets
    return X, y, classes


# ---- Landmark normalization functions ----

def get_center_point(landmarks, left, right):
    """
    Returns the midpoint between two landmarks.
    Used to define the center of hips or shoulders.
    """
    l = tf.gather(landmarks, left.value, axis=1)
    r = tf.gather(landmarks, right.value, axis=1)
    return (l + r) * 0.5


def get_pose_size(landmarks, m=2.5):
    """
    Computes a scale factor for the pose:
    - Uses torso length (shoulders to hips)
    - Also considers max distance from pose center to any keypoint
    - Final size = max(torso * m, max_dist)
    """
    hips = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    shoulders = get_center_point(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)
    torso_sz = tf.norm(shoulders - hips, axis=-1)

    center = tf.expand_dims(hips, 1)
    center = tf.broadcast_to(center, tf.shape(landmarks))
    dists = tf.norm(landmarks - center, axis=2)
    maxd = tf.reduce_max(dists, axis=1)

    return tf.maximum(torso_sz * m, maxd)


def normalize_landmarks(landmarks):
    """
    Normalizes landmarks:
    - Translates landmarks so that center is at origin
    - Scales so pose fits within unit size
    """
    center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    center = tf.expand_dims(center, 1)
    center = tf.broadcast_to(center, tf.shape(landmarks))

    lm = landmarks - center
    size = tf.reshape(get_pose_size(lm), [-1, 1, 1])
    return lm / size


def to_embedding(flat):
    """
    Converts raw landmark array (flat 51-length) to a normalized 34-length embedding:
    - Reshapes into (17, 3) format
    - Keeps only x, y
    - Normalizes, then flattens back to 34D
    """
    reshaped = tf.reshape(flat, (-1, 17, 3))
    coords = reshaped[:, :, :2]
    normed = normalize_landmarks(coords)
    return tf.reshape(normed, (-1, 34))


# ---- Training pipeline ----

if __name__ == "__main__":
    # Load data
    X, y, classes = load_csv(TRAIN_CSV)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    X_test, y_test, _ = load_csv(TEST_CSV)

    # Normalize and transform into embeddings
    emb_train = to_embedding(tf.convert_to_tensor(X_train))
    emb_val   = to_embedding(tf.convert_to_tensor(X_val))
    emb_test  = to_embedding(tf.convert_to_tensor(X_test))

    # ---- Define neural network ----
    # Input: 34D embedding (normalized x/y for 17 landmarks)
    inp = keras.Input(shape=(34,))
    x = keras.layers.Dense(128, activation='relu')(inp)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    out = keras.layers.Dense(len(classes), activation='softmax')(x)

    model = keras.Model(inp, out)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # ---- Callbacks ----
    # Save best model during training
    ckpt = keras.callbacks.ModelCheckpoint('best_model.h5',
                                           monitor='val_accuracy',
                                           save_best_only=True)

    # Stop training early if validation accuracy doesn't improve
    es = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                       patience=20,
                                       restore_best_weights=True)

    # ---- Train the model ----
    model.fit(emb_train, y_train,
              validation_data=(emb_val, y_val),
              epochs=200,
              batch_size=16,
              callbacks=[ckpt, es])

    # ---- Evaluate on test set ----
    loss, acc = model.evaluate(emb_test, y_test)
    print(f"Test loss: {loss:.4f}, accuracy: {acc:.4f}")

    # ---- Save the final model ----
    model.save(MODEL_H5)
    print(f"Saved Keras model to {MODEL_H5}")
