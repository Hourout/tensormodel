from typing import Optional, Tuple

import cv2
import numpy as np
import linora as la
import tensorflow as tf
from tensorflow.keras import layers

__all__ = ['NSFW_Model']


def _batch_norm(name: str) -> layers.BatchNormalization:
    return layers.BatchNormalization(name=name, epsilon=1e-05)

def _conv_block(stage: int, block: int, inputs: tf.Tensor, nums_filters: Tuple[int, int, int],
                kernel_size: int = 3, stride: int = 2) -> tf.Tensor:
    num_filters_1, num_filters_2, num_filters_3 = nums_filters

    conv_name_base = f"conv_stage{stage}_block{block}_branch"
    bn_name_base = f"bn_stage{stage}_block{block}_branch"
    shortcut_name_post = f"_stage{stage}_block{block}_proj_shortcut"

    shortcut = layers.Conv2D(name=f"conv{shortcut_name_post}", filters=num_filters_3, kernel_size=1, strides=stride, padding="same")(inputs)
    shortcut = _batch_norm(f"bn{shortcut_name_post}")(shortcut)

    x = layers.Conv2D(name=f"{conv_name_base}2a", filters=num_filters_1, kernel_size=1, strides=stride, padding="same")(inputs)
    x = _batch_norm(f"{bn_name_base}2a")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(name=f"{conv_name_base}2b", filters=num_filters_2, kernel_size=kernel_size, strides=1, padding="same")(x)
    x = _batch_norm(f"{bn_name_base}2b")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(name=f"{conv_name_base}2c", filters=num_filters_3, kernel_size=1, strides=1, padding="same")(x)
    x = _batch_norm(f"{bn_name_base}2c")(x)
    x = layers.Add()([x, shortcut])
    return layers.Activation("relu")(x)


def _identity_block(stage: int, block: int, inputs: tf.Tensor, 
                    nums_filters: Tuple[int, int, int], kernel_size: int) -> tf.Tensor:
    num_filters_1, num_filters_2, num_filters_3 = nums_filters
    conv_name_base = f"conv_stage{stage}_block{block}_branch"
    bn_name_base = f"bn_stage{stage}_block{block}_branch"

    x = layers.Conv2D(name=f"{conv_name_base}2a", filters=num_filters_1, kernel_size=1, strides=1, padding="same")(inputs)
    x = _batch_norm(f"{bn_name_base}2a")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(name=f"{conv_name_base}2b", filters=num_filters_2, kernel_size=kernel_size, strides=1, padding="same")(x)
    x = _batch_norm(f"{bn_name_base}2b")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(name=f"{conv_name_base}2c", filters=num_filters_3, kernel_size=1, strides=1, padding="same")(x)
    x = _batch_norm(f"{bn_name_base}2c")(x)
    x = layers.Add()([x, inputs])
    return layers.Activation("relu")(x)


def make_open_nsfw_model(input_shape=(224, 224, 3)):
    image_input = layers.Input(shape=input_shape, name="input")
    x = image_input
    x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
    x = layers.Conv2D(name="conv_1", filters=64, kernel_size=7, strides=2, padding="valid")(x)
    x = _batch_norm("bn_1")(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    x = _conv_block(stage=0, block=0, inputs=x, nums_filters=(32, 32, 128), kernel_size=3, stride=1)
    x = _identity_block(stage=0, block=1, inputs=x, nums_filters=(32, 32, 128), kernel_size=3)
    x = _identity_block(stage=0, block=2, inputs=x, nums_filters=(32, 32, 128), kernel_size=3)

    x = _conv_block(stage=1, block=0, inputs=x, nums_filters=(64, 64, 256), kernel_size=3, stride=2)
    x = _identity_block(stage=1, block=1, inputs=x, nums_filters=(64, 64, 256), kernel_size=3)
    x = _identity_block(stage=1, block=2, inputs=x, nums_filters=(64, 64, 256), kernel_size=3)
    x = _identity_block(stage=1, block=3, inputs=x, nums_filters=(64, 64, 256), kernel_size=3)

    x = _conv_block(stage=2, block=0, inputs=x, nums_filters=(128, 128, 512), kernel_size=3, stride=2)
    x = _identity_block(stage=2, block=1, inputs=x, nums_filters=(128, 128, 512), kernel_size=3)
    x = _identity_block(stage=2, block=2, inputs=x, nums_filters=(128, 128, 512), kernel_size=3)
    x = _identity_block(stage=2, block=3, inputs=x, nums_filters=(128, 128, 512), kernel_size=3)
    x = _identity_block(stage=2, block=4, inputs=x, nums_filters=(128, 128, 512), kernel_size=3)
    x = _identity_block(stage=2, block=5, inputs=x, nums_filters=(128, 128, 512), kernel_size=3)

    x = _conv_block(stage=3, block=0, inputs=x, nums_filters=(256, 256, 1024), kernel_size=3, stride=2)
    x = _identity_block(stage=3, block=1, inputs=x, nums_filters=(256, 256, 1024), kernel_size=3)
    x = _identity_block(stage=3, block=2, inputs=x, nums_filters=(256, 256, 1024), kernel_size=3)

    x = layers.AveragePooling2D(pool_size=7, strides=1, padding="valid", name="pool")(x)
    x = layers.Flatten()(x)
    logits = layers.Dense(name="fc_nsfw", units=2)(x)
    output = layers.Activation("softmax", name="predictions")(logits)
    model = tf.keras.Model(image_input, output)
    return model

def preprocess_image(image):
    if isinstance(image, str):
        image = la.image.read_image(image)
    image = la.image.color_convert(image, la.image.ColorMode.RGB)
    image = la.image.resize(image, (224, 224), la.image.ResizeMethod.BILINEAR)
    image = la.image.image_to_array(image)
    image = image[:, :, ::-1]-np.array([104., 117., 123.], dtype=np.float32)
    return image

class NSFW_Model():
    def __init__(self, weights_path, gpu_strategy=None):
        if not la.gfile.exists(weights_path):
            url = 'https://github.com/bhky/opennsfw2/releases/download/v0.1.0/open_nsfw_weights.h5'
            la.data.get_file(url, weights_path)
        if gpu_strategy is not None:
            gpu_strategy()
        self.model = make_open_nsfw_model()
        self.model.load_weights(weights_path)
        
    def predict_images(self, image_file, batch_size=None):
        if isinstance(image_file, str):
            image_file = [image_file]
        images = np.array([preprocess_image(image_path) for image_path in image_file])
        predictions = self.model.predict(images, batch_size=len(images) if batch_size is None else batch_size)
        return [round(i, 3) for i in predictions[:,1].tolist()]

    def predict_video(self, video_path, frame_interval=8, prob_threshold=0.8, output_video_path=None):
        """Make prediction for each video frame."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_writer = None

        nsfw_prob = 0.0
        nsfw_probs = []
        frame_count = 0

        while cap.isOpened():
            ret, bgr_frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            if video_writer is None and output_video_path is not None:
                video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                                               fps, (frame.shape[1], frame.shape[0]))

            if frame_count == 1 or (frame_count + 1) % frame_interval == 0:
                pil_frame = la.image.array_to_image(frame)
                input_frame = preprocess_image(pil_frame)
                nsfw_prob = self.model.predict(np.expand_dims(input_frame, axis=0), 1).round(3)[0][1]
                nsfw_probs.extend([nsfw_prob]*frame_interval)

            if video_writer is not None:
                result_text = f"NSFW probability: {nsfw_prob:.3f}"
                colour = (255, 0, 0) if nsfw_prob >= prob_threshold else (0, 0, 255)
                cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2, cv2.LINE_AA)
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if video_writer is not None:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        elapsed_seconds = (np.arange(1, len(nsfw_probs) + 1) / fps).round(2).tolist()
        return {'frame_time':elapsed_seconds, 'frame_prob':nsfw_probs}

