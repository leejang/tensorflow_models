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

"""Class predictor for image-level classification

"""
import tensorflow as tf
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import static_shape

slim = tf.contrib.slim

IMAGE_LEVEL_CLASS_PREDICTIONS = 'image_level_class_predictions'


class ImageLevelConvolutionalClassPredictor(object):
  """Image-level Convolutional Class Predictor.

  Optionally add an intermediate 1x1 convolutional layer after features and
  predict in parallel branches box_encodings and
  class_predictions_with_background.

  Currently this box predictor assumes that predictions are "shared" across
  classes --- that is each anchor makes box predictions which do not depend
  on class.
  """

  def __init__(self,
               is_training,
               num_classes,
               conv_hyperparams,
               use_dropout,
               dropout_keep_prob,
               kernel_size,
               class_prediction_bias_init=0.0,
               apply_sigmoid_to_scores=False):
    """Constructor.

    Args:
      is_training: Indicates whether the BoxPredictor is in training mode.
      num_classes: number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
      conv_hyperparams: Slim arg_scope with hyperparameters for convolution ops.
      use_dropout: Option to use dropout for class prediction or not.
      dropout_keep_prob: Keep probability for dropout.
        This is only used if use_dropout is True.
      kernel_size: Size of final convolution kernel.  If the
        spatial resolution of the feature map is smaller than the kernel size,
        then the kernel size is automatically set to be
        min(feature_width, feature_height).
      apply_sigmoid_to_scores: if True, apply the sigmoid on the output
        class_predictions.
      class_prediction_bias_init: constant value to initialize bias of the last
        conv2d layer before class prediction.

    """

    self._is_training = is_training
    self._num_classes = num_classes
    self._conv_hyperparams = conv_hyperparams
    self._use_dropout = use_dropout
    self._kernel_size = kernel_size
    self._dropout_keep_prob = dropout_keep_prob
    self._class_prediction_bias_init = class_prediction_bias_init
    self._apply_sigmoid_to_scores = apply_sigmoid_to_scores

  #def predict(self, image_features, class_predictor_scope):
  def predict(self, image_features, audio_features, class_predictor_scope):
    """Computes encoded object locations and corresponding confidences.

    Args:
      image_features: A float tensor of shape [batch_size, height, width,
        channels] containing features for a batch of images.

    Returns:
      A dictionary containing the following tensors.
        image_level_class_predictions: A float tensor of shape
          [batch_size, num_classes] representing the class
          predictions for the proposals.
    """

    num_class_slots = self._num_classes + 1
    #num_class_slots = self._num_classes
    num_hidden_neurons = 256
    net = image_features
    net = tf.concat([image_features, audio_features], 3)

    print("image_features", image_features.get_shape())
    print("audio_features", audio_features.get_shape())
    print("concat_net_features", net.get_shape())

    with tf.variable_scope(class_predictor_scope):
        with slim.arg_scope(self._conv_hyperparams), \
            slim.arg_scope([slim.dropout], is_training=self._is_training):
            with slim.arg_scope([slim.conv2d], activation_fn=None,
                                normalizer_fn=None, normalizer_params=None):
             
                if self._use_dropout:
                    net = slim.dropout(net, keep_prob=self._dropout_keep_prob)
               
                """ 
                image_level_class_predictions = slim.conv2d(
                    net,  num_hidden_neurons,
                    [self._kernel_size, self._kernel_size],  scope=class_predictor_scope,
                    biases_initializer=tf.constant_initializer(self._class_prediction_bias_init)
                    )
                """

                image_level_class_predictions = slim.conv2d(
                    net,  num_hidden_neurons,
                    [self._kernel_size, self._kernel_size], padding="VALID", scope=class_predictor_scope,
                    biases_initializer=tf.constant_initializer(self._class_prediction_bias_init)
                    )
 
                if self._apply_sigmoid_to_scores:
                    image_level_class_predictions = tf.sigmoid(image_level_class_predictions)
  

                combined_feature_map_shape = shape_utils.combined_static_and_dynamic_shape(image_features)

                """
                image_level_class_predictions = slim.conv2d(
                    image_level_class_predictions,  num_hidden_neurons/2,
                    [2, 2], padding="VALID", biases_initializer=tf.constant_initializer(self._class_prediction_bias_init)
                    )

                image_level_class_predictions = slim.conv2d(
                    image_level_class_predictions,  num_class_slots,
                    [1, 1], padding="VALID", biases_initializer=tf.constant_initializer(self._class_prediction_bias_init)
                    )
                """

                print("kernel_size", self._kernel_size)
                # batch x 1 x 1 x 2048
                print("image_level_class_predictions", image_level_class_predictions.get_shape())
                #print("audio_features", audio_features.get_shape())

                # flatten to use fc layer
                image_level_class_predictions = slim.flatten(image_level_class_predictions)
                #audio_features = slim.flatten(audio_features)

                #concat_features = tf.concat([image_level_class_predictions, audio_features], 1)

                #print("concat_features", concat_features.get_shape())

                #image_level_class_predictions = slim.fully_connected(concat_features,
                #                                                      num_hidden_neurons)

                # fully connected
                image_level_class_predictions = slim.fully_connected(image_level_class_predictions,
                                                                      num_class_slots)
                # re-shape
                image_level_class_predictions = tf.reshape(
                    image_level_class_predictions,
                    tf.stack([combined_feature_map_shape[0],
                              num_class_slots]))

                print("image_level_class_predictions", image_level_class_predictions.get_shape())

    return {IMAGE_LEVEL_CLASS_PREDICTIONS:image_level_class_predictions}
