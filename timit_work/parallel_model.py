"""
Multi-GPU Support for Keras 2.2.4.

Copyright (c) 2018 EastDawn, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Ziyigogogo

Make the keras official multi-gpu more usable as the Matterport's
while also making use of the "inplace split tech" from official code
achieving faster speed & gpu memory efficiency


Ideas and a code snippets from these sources:
https://github.com/keras-team/keras/blob/master/keras/utils/multi_gpu_utils.py
https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/parallel_model.py

"""

import time

import keras.backend as K
import tensorflow as tf
from keras.layers.core import Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras.utils.generic_utils import to_list


class MultiGpuModel(Model):

    def __init__(self, template_model, num_gpus):
        super(MultiGpuModel, self).__init__(**self.get_inputs_and_outputs(template_model, num_gpus))
        self.inner_model = template_model
        self.num_gpus = num_gpus

    def __getattribute__(self, attrname):
        """Redirect loading and saving methods to the inner model. That's where
        the weights are stored."""
        if 'load' in attrname or 'save' in attrname:
            return getattr(self.inner_model, attrname)
        return super(MultiGpuModel, self).__getattribute__(attrname)

    def summary(self, *args, **kwargs):
        """Override summary() to display summaries of both, the wrapper and inner models."""
        super(MultiGpuModel, self).summary(*args, **kwargs)
        self.inner_model.summary(*args, **kwargs)

    def get_inputs_and_outputs(self, template_model, num_gpus):
        print("start preparing multi-gpu model on {} gpus...".format(num_gpus))
        tic = time.time()

        available_devices = [self._normalize_device_name(name) for name in self._get_available_devices()]
        target_gpu_ids = range(num_gpus)
        target_devices = ['/cpu:0'] + ['/gpu:{}'.format(i) for i in target_gpu_ids]

        for device in target_devices:
            if device not in available_devices:
                raise ValueError(
                    'To call `ParallelModel` with `num_gpus={}`, '
                    'we expect the following devices to be available: {}. '
                    'However this machine only has: {}. '
                    'Try reducing `num_gpus`.'.format(num_gpus, target_devices, available_devices)
                )

        all_outputs = [[] for _ in range(len(template_model.outputs))]
        # Place a copy of the model on each GPU, each getting a slice of the inputs.
        for i, gpu_id in enumerate(target_gpu_ids):
            with tf.device('/gpu:{}'.format(gpu_id)):
                with tf.name_scope('replica_{}'.format(gpu_id)):
                    inputs = []
                    # Retrieve a slice of the input.
                    for x in template_model.inputs:
                        # In-place input splitting 5% ~ 12% faster and also less GPU memory duplication.
                        with tf.device(x.device):
                            input_shape = K.int_shape(x)[1:]
                            slice_i = Lambda(
                                self.get_slice,
                                output_shape=input_shape,
                                arguments={'i': i, 'parts': num_gpus}
                            )(x)
                            inputs.append(slice_i)

                    # Apply model on slice
                    # (creating a model replica on the target device).
                    outputs = template_model(inputs)
                    outputs = to_list(outputs)

                    # Save the outputs for merging back together later.
                    for o in range(len(outputs)):
                        all_outputs[o].append(outputs[o])

        # Deduplicate output names to handle Siamese networks.
        occurrences = {}
        for n in template_model.output_names:
            if n not in occurrences:
                occurrences[n] = 1
            else:
                occurrences[n] += 1
        conflict_counter = {n: 0 for n, count in occurrences.items() if count > 1}
        output_names = []
        for n in template_model.output_names:
            if n in conflict_counter:
                conflict_counter[n] += 1
                n += '_{}'.format(conflict_counter[n])
            output_names.append(n)

        with tf.device('/cpu:0'):
            merged = []
            for name, outputs in zip(output_names, all_outputs):

                if K.int_shape(outputs[0]) == ():
                    # Average
                    m = Lambda(lambda output: tf.add_n(output) / len(outputs), name=name)(outputs)
                else:
                    # Concatenate
                    m = concatenate(outputs, axis=0, name=name)
                merged.append(m)

        toc = time.time()
        print("multi-gpu model done! Cost: {}s".format(toc - tic))

        return {
            'inputs': template_model.inputs,
            'outputs': merged
        }

    @staticmethod
    def _get_available_devices():
        return [x.name for x in K.get_session().list_devices()]

    @staticmethod
    def _normalize_device_name(name):
        name = '/' + ':'.join(name.lower().replace('/', '').split(':')[-2:])
        return name

    @staticmethod
    def get_slice(data, i, parts):
        shape = K.shape(data)
        batch_size = shape[:1]
        input_shape = shape[1:]
        step = batch_size // parts
        if i == parts - 1:
            size = batch_size - step * i
        else:
            size = step
        size = K.concatenate([size, input_shape], axis=0)
        stride = K.concatenate([step, input_shape * 0], axis=0)
        start = stride * i
        return K.slice(data, start, size)