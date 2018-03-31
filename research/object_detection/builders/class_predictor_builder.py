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

"""Function to build class predictor from configuration."""

from object_detection.core import class_predictor
from object_detection.protos import class_predictor_pb2


def build(argscope_fn, class_predictor_config, is_training):
  """Builds class predictor based on the configuration.

  Builds class predictor based on the configuration. See class_predictor.proto for
  configurable options. Also, see class_predictor.py for more details.

  Args:
    argscope_fn: A function that takes the following inputs:
        * hyperparams_pb2.Hyperparams proto
        * a boolean indicating if the model is in training mode.
      and returns a tf slim argscope for Conv and FC hyperparameters.
    class_predictor_config: box_predictor_pb2.BoxPredictor proto containing
      configuration.
    is_training: Whether the models is in training mode.
    num_classes: Number of classes to predict.

  Returns:
    class_predictor: box_predictor.BoxPredictor object.

  Raises:
    ValueError: On unknown box predictor.
  """

  num_classes = class_predictor_config.num_classes

  if not isinstance(class_predictor_config, class_predictor_pb2.ClassPredictor):
    raise ValueError('class_predictor_config not of type '
                     'class_predictor_pb2.ClassPredictor.')

  class_predictor_oneof = class_predictor_config.WhichOneof('class_predictor_oneof')

  if class_predictor_oneof == 'image_level_convolutional_class_predictor':
    conv_class_predictor = class_predictor_config.image_level_convolutional_class_predictor
    conv_hyperparams = argscope_fn(conv_class_predictor.conv_hyperparams,
                                   is_training)
    class_predictor_object = class_predictor.ImageLevelConvolutionalClassPredictor(
        is_training=is_training,
        num_classes=num_classes,
        conv_hyperparams=conv_hyperparams,
        use_dropout=conv_class_predictor.use_dropout,
        dropout_keep_prob=conv_class_predictor.dropout_keep_probability,
        kernel_size=conv_class_predictor.kernel_size,
        apply_sigmoid_to_scores=conv_class_predictor.apply_sigmoid_to_scores,
        class_prediction_bias_init=conv_class_predictor.class_prediction_bias_init
    )
    return class_predictor_object

  raise ValueError('Unknown class predictor: {}'.format(class_predictor_oneof))
