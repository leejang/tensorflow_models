# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for object_detection.trainer."""

import tensorflow as tf

from google.protobuf import text_format

from object_detection import trainer
from object_detection.core import losses
from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.core import class_predictor
from object_detection.protos import train_pb2
from object_detection.protos import hyperparams_pb2
from object_detection.builders import hyperparams_builder


NUMBER_OF_CLASSES = 2
NUMBER_OF_CLASSES_IN_IMAGE_LEVEL = 7


def get_input_function():
  """A function to get test inputs. Returns an image with one box."""
  image = tf.random_uniform([32, 32, 3], dtype=tf.float32)
  key = tf.constant('image_000000')
  class_label = tf.random_uniform(
      [1], minval=0, maxval=NUMBER_OF_CLASSES, dtype=tf.int32)
  # first element is a shape of the tensor
  image_class_labels = tf.random_uniform(
      [2], minval=0, maxval=NUMBER_OF_CLASSES_IN_IMAGE_LEVEL, dtype=tf.int32)
  box_label = tf.random_uniform(
      [1, 4], minval=0.4, maxval=0.6, dtype=tf.float32)

  return {
      fields.InputDataFields.image: image,
      fields.InputDataFields.key: key,
      fields.InputDataFields.groundtruth_classes: class_label,
      fields.InputDataFields.groundtruth_boxes: box_label,
      fields.InputDataFields.groundtruth_image_classes: image_class_labels
  }


class FakeDetectionModel(model.DetectionModel):
  """A simple (and poor) DetectionModel for use in test."""

  def __init__(self):
    super(FakeDetectionModel, self).__init__(num_classes=NUMBER_OF_CLASSES)

    conv_hyperparams_text_proto = """
      regularizer {
        l1_regularizer {
          weight: 0.0003
        }
      }
      initializer {
        truncated_normal_initializer {
          mean: 0.0
          stddev: 0.3
        }
      }
      activation: RELU_6
    """
    hyperparams_proto = hyperparams_pb2.Hyperparams()
    text_format.Merge(conv_hyperparams_text_proto, hyperparams_proto)

    self._class_predictor = class_predictor.ImageLevelConvolutionalClassPredictor(
        is_training=False,
        num_classes=NUMBER_OF_CLASSES_IN_IMAGE_LEVEL,
        conv_hyperparams=hyperparams_proto,
        use_dropout=True,
        dropout_keep_prob=0.5,
        kernel_size=5,
        class_prediction_bias_init=0.0,
        apply_sigmoid_to_scores=False)

    self._classification_loss = losses.WeightedSigmoidClassificationLoss(
        anchorwise_output=True)
    self._localization_loss = losses.WeightedSmoothL1LocalizationLoss(
        anchorwise_output=True)
    self._classification_in_image_level_loss = losses.WeightedSigmoidClassificationLossInImageLevel() 


  def preprocess(self, inputs):
    """Input preprocessing, resizes images to 28x28.

    Args:
      inputs: a [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: a [batch, 28, 28, channels] float32 tensor.
    """
    return tf.image.resize_images(inputs, [28, 28])

  def predict(self, preprocessed_inputs):
    """Prediction tensors from inputs tensor.

    Args:
      preprocessed_inputs: a [batch, 28, 28, channels] float32 tensor.

    Returns:
      prediction_dict: a dictionary holding prediction tensors to be
        passed to the Loss or Postprocess functions.
    """
    flattened_inputs = tf.contrib.layers.flatten(preprocessed_inputs)
    class_prediction = tf.contrib.layers.fully_connected(
        flattened_inputs, self._num_classes)
    box_prediction = tf.contrib.layers.fully_connected(flattened_inputs, 4)
    class_predictions_in_image_level = tf.contrib.layers.fully_connected(
        flattened_inputs, self._class_predictor._num_classes)
 
    return {
        'class_predictions_with_background': tf.reshape(
            class_prediction, [-1, 1, self._num_classes]),
        'box_encodings': tf.reshape(box_prediction, [-1, 1, 4]),
        'class_predictions_in_image_level': class_predictions_in_image_level
    }

  def postprocess(self, prediction_dict, **params):
    """Convert predicted output tensors to final detections. Unused.

    Args:
      prediction_dict: a dictionary holding prediction tensors.
      **params: Additional keyword arguments for specific implementations of
        DetectionModel.

    Returns:
      detections: a dictionary with empty fields.
    """
    return {
        'detection_boxes': None,
        'detection_scores': None,
        'detection_classes': None,
        'num_detections': None,
        'detection_scores_in_image_level': None,
        'detection_classes_in_image_level': None
    }

  def loss(self, prediction_dict):
    """Compute scalar loss tensors with respect to provided groundtruth.

    Calling this function requires that groundtruth tensors have been
    provided via the provide_groundtruth function.

    Args:
      prediction_dict: a dictionary holding predicted tensors

    Returns:
      a dictionary mapping strings (loss names) to scalar tensors representing
        loss values.
    """
    batch_reg_targets = tf.stack(
        self.groundtruth_lists(fields.BoxListFields.boxes))
    batch_cls_targets = tf.stack(
        self.groundtruth_lists(fields.BoxListFields.classes))
    batch_cls_targets_in_image_level = tf.stack(
        self.groundtruth_lists(fields.BoxListFields.image_level_classes))
    weights = tf.constant(
        1.0, dtype=tf.float32,
        shape=[len(self.groundtruth_lists(fields.BoxListFields.boxes)), 1])

    location_losses = self._localization_loss(
        prediction_dict['box_encodings'], batch_reg_targets,
        weights=weights)
    cls_losses = self._classification_loss(
        prediction_dict['class_predictions_with_background'], batch_cls_targets,
        weights=weights)

    cls_losses_in_image_level = self._classification_in_image_level_loss(
          prediction_dict['class_predictions_in_image_level'],
          batch_cls_targets_in_image_level)


    loss_dict = {
        'localization_loss': tf.reduce_sum(location_losses),
        'classification_loss': tf.reduce_sum(cls_losses),
        'classification_loss_in_image_level': tf.reduce_sum(cls_losses_in_image_level)
    }
    return loss_dict

  def restore_map(self, from_detection_checkpoint=True):
    """Returns a map of variables to load from a foreign checkpoint.

    Args:
      from_detection_checkpoint: whether to restore from a full detection
        checkpoint (with compatible variable names) or to restore from a
        classification checkpoint for initialization prior to training.

    Returns:
      A dict mapping variable names to variables.
    """
    return {var.op.name: var for var in tf.global_variables()}


class TrainerTest(tf.test.TestCase):

  def test_configure_trainer_and_train_two_steps(self):
    train_config_text_proto = """
    optimizer {
      adam_optimizer {
        learning_rate {
          constant_learning_rate {
            learning_rate: 0.01
          }
        }
      }
    }
    data_augmentation_options {
      random_adjust_brightness {
        max_delta: 0.2
      }
    }
    data_augmentation_options {
      random_adjust_contrast {
        min_delta: 0.7
        max_delta: 1.1
      }
    }
    num_steps: 2
    """
    train_config = train_pb2.TrainConfig()
    text_format.Merge(train_config_text_proto, train_config)

    train_dir = self.get_temp_dir()

    trainer.train(create_tensor_dict_fn=get_input_function,
                  create_model_fn=FakeDetectionModel,
                  train_config=train_config,
                  master='',
                  task=0,
                  num_clones=1,
                  worker_replicas=1,
                  clone_on_cpu=True,
                  ps_tasks=0,
                  worker_job_name='worker',
                  is_chief=True,
                  train_dir=train_dir)


if __name__ == '__main__':
  tf.test.main()
