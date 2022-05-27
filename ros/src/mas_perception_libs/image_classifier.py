import os
import yaml
from abc import ABCMeta, abstractmethod
import numpy as np
from cv_bridge import CvBridge
from mas_perception_libs.utils import get_classes_in_data_dir, process_image_message


class ImageClassifier(object):
    """
    Abstract class for different models of image classifier
    """
    __metaclass__ = ABCMeta

    _classes = None     # type: dict

    def __init__(self, **kwargs):
        # read information on classes, either directly, via a file, or from a data directory
        self._classes = kwargs.get('classes', None)

        if self._classes is None:
            class_file = kwargs.get('class_file', None)
            if class_file is not None and os.path.exists(class_file):
                with open(class_file) as infile:
                    if yaml.__version__ < '5.1':
                        self._classes = yaml.load(infile)
                    else:
                        self._classes = yaml.load(infile, Loader=yaml.FullLoader)

        if self._classes is None:
            data_dir = kwargs.get('data_dir', None)
            if data_dir is None:
                raise ValueError('no class definition specified')
            if not os.path.exists(data_dir):
                raise ValueError('Directory does not exist: ' + data_dir)

            self._classes = get_classes_in_data_dir(data_dir)

    @property
    def classes(self):
        """
        dictionary mapping from predicted numeric class value to class name
        """
        return self._classes

    @abstractmethod
    def classify(self, image_messages):
        """
        method to be implemented by extensions TODO(minhnh) refactor preprocessing to base class

        :param image_messages: list of sensor_msgs/Image messages
        :return: (indices, predicted_classes, confidences), where:
                 - indices: image indices of the input list, track for which images the predictions are
                 - predicted_classes: predicted class names
                 - confidences: prediction confidences
        :rtype: tuple
        """
        pass


class ImageClassifierTest(ImageClassifier):
    """
    Extension of ImageClassifier for testing, return random classes
    """
    def __init__(self, **kwargs):
        super(ImageClassifierTest, self).__init__(**kwargs)

    def classify(self, image_messages):
        import random
        indices = list(range(len(image_messages)))
        classes = [self.classes[random.randint(0, len(self.classes) - 1)] for _ in indices]
        probabilities = [random.random() for _ in indices]
        return indices, classes, probabilities


class KerasImageClassifier(ImageClassifier):
    """
    Extension of ImageClassifier for models implemented using Keras
    """
    def __init__(self, **kwargs):
        from keras.models import Model, load_model

        super(KerasImageClassifier, self).__init__(**kwargs)

        self._model = kwargs.get('model', None)
        model_path = kwargs.get('model_path', None)
        if self._model is None:
            if model_path is not None:
                self._model = load_model(model_path)
                # see https://github.com/keras-team/keras/issues/6462
                self._model._make_predict_function()
            else:
                raise ValueError('No model object or path passed received')

        if not isinstance(self._model, Model):
            raise ValueError('model is not a Keras Model object')

        if len(self._classes) != self._model.output_shape[-1]:
            raise ValueError('number of classes ({0}) does not match model output shape ({1})'
                             .format(len(self._classes), self._model.output_shape[-1]))

        self._img_preprocess_func = kwargs.get('img_preprocess_func', None)

        # assume input shape is 3D with channel dimension to be 3
        self._target_size = tuple(i for i in self._model.input_shape if i != 3 and i is not None)

        # CvBridge for ROS image conversion
        self._cv_bridge = CvBridge()

    def classify_np_images(self, np_images):
        """
        Classify NumPy images
        """
        image_tensor = []
        indices = []
        for i in range(len(np_images)):
            if np_images[i] is None:
                # skip broken images
                continue

            image_tensor.append(np_images[i])
            indices.append(i)

        image_tensor = np.array(image_tensor)
        preds = self._model.predict(image_tensor)
        class_indices = np.argmax(preds, axis=1)
        confidences = np.max(preds, axis=1)
        predicted_classes = [self._classes[i] for i in class_indices]

        return indices, predicted_classes, confidences

    def classify(self, image_messages):
        """
        Classify ROS `sensor_msgs/Image` messages
        """
        if len(image_messages) == 0:
            return [], [], []

        np_images = [process_image_message(msg, self._cv_bridge, self._target_size, self._img_preprocess_func)
                     for msg in image_messages]

        return self.classify_np_images(np_images)
