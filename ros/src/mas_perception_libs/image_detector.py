from importlib import import_module

import os
from abc import ABCMeta, abstractmethod
import yaml
from enum import Enum
import numpy as np

import rospy
from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge

from .utils import process_image_message
from .bounding_box import BoundingBox2D
from .visualization import bgr_dict_from_classes, draw_labeled_boxes_img_msg


class ImageDetectionKey(Enum):
    CLASS = 0
    CONF = 1
    X_MIN = 2
    X_MAX = 3
    Y_MIN = 4
    Y_MAX = 5


class ImageDetectorBase(object):
    """
    Abstract class for detecting things in images
    """
    __metaclass__ = ABCMeta

    _classes = None                 # type: dict
    _class_colors = None            # type: dict
    _cv_bridge = None               # type: CvBridge
    # input size of model, will be ignored if left None
    _target_size = None             # type: tuple
    # preprocess function for each input image, will be ignored if left None
    _img_preprocess_func = None     # type: function

    def __init__(self, **kwargs):
        # for ROS image message conversion
        self._cv_bridge = CvBridge()

        # load dictionary of classes
        self._classes = kwargs.get('classes', None)
        if self._classes is None:
            class_file = kwargs.get('class_file', None)
            if class_file is not None and os.path.exists(class_file):
                with open(class_file, 'r') as infile:
                    self._classes = yaml.load(infile, Loader=yaml.SafeLoader)

        if self._classes is None:
            raise ValueError("no valid 'class_file' or 'classes' parameter specified")

        # generate colors for each class for visualization
        self._class_colors = bgr_dict_from_classes(list(self._classes.values()))

        # load kwargs file and call load_model()
        model_kwargs_file = kwargs.get('model_kwargs_file', None)
        if model_kwargs_file is not None and os.path.exists(model_kwargs_file):
            with open(model_kwargs_file, 'r') as infile:
                load_model_kwargs = yaml.load(infile, Loader=yaml.SafeLoader)
        else:
            load_model_kwargs = {}

        self.load_model(**load_model_kwargs)

    @property
    def classes(self):
        """ dictionary which maps prediction value (int) to class name (str) """
        return self._classes

    @property
    def class_colors(self):
        """ dictionary which maps from class name (str) to RGB colors (3-tuple) """
        return self._class_colors

    @abstractmethod
    def load_model(self, **kwargs):
        """
        To be implemented by extensions, where detection model is loaded

        :param kwargs: key word arguments necessary for the detection model
        :return: None
        """
        pass

    @abstractmethod
    def _detect(self, np_images, orig_img_sizes):
        """
        To be implemented by extensions, detect objects in given image messages

        :param np_images: list of numpy images extracted from image messages
        :param orig_img_sizes: list of original images' (width, height), necessary to map detected bounding boxes back
                               to the original images if the images are resized to fit the detection model input
        :return: List of predictions for each image. Each prediction is a list of dictionaries representing the detected
                 classes with their bounding boxes and confidences. The dictionary keys are values of the
                 ImageDetectionKey Enum.
        """
        pass

    def detect(self, image_messages):
        """
        Preprocess image messages then call abstract method _detect() on the processed images

        :param image_messages: list of sensor_msgs/Image
        :return: same with _detect()
        """
        if len(image_messages) == 0:
            return []

        np_images = []
        orig_img_sizes = []
        for msg in image_messages:
            np_images.append(process_image_message(msg, self._cv_bridge, self._target_size, self._img_preprocess_func))
            orig_img_sizes.append((msg.width, msg.height))

        return self._detect(np_images, orig_img_sizes)

    def visualize_detection(self, img_msg, bounding_boxes):
        """
        Draw detected classes on an image message

        :param img_msg: sensor_msgs/Image message to be drawn on
        :param bounding_boxes: list of BoundingBox2D objects created from prediction
        :return: sensor_msgs/Image message with detected boxes drawn on top
        """
        return draw_labeled_boxes_img_msg(self._cv_bridge, img_msg, bounding_boxes)

    @staticmethod
    def prediction_to_bounding_boxes(prediction, color_dict=None):
        """
        Create BoundingBox2D objects from a prediction result

        :param prediction: List of dictionaries representing detected classes in an image. Keys are values of
                           ImageDetectionKey Enum
        :param color_dict: Dictionary mapping class name to a color tuple (r, g, b). Default color is blue.
        :return: List of BoundingBox2D objects, one for each predicted class
        """
        boxes = []
        classes = []
        confidences = []
        for box_dict in prediction:
            box_geometry = (box_dict[ImageDetectionKey.X_MIN],
                            box_dict[ImageDetectionKey.Y_MIN],
                            box_dict[ImageDetectionKey.X_MAX] - box_dict[ImageDetectionKey.X_MIN],
                            box_dict[ImageDetectionKey.Y_MAX] - box_dict[ImageDetectionKey.Y_MIN])

            label = '{}: {:.2f}'.format(box_dict[ImageDetectionKey.CLASS], box_dict[ImageDetectionKey.CONF])

            if color_dict is None:
                color = (0, 0, 255)     # default color: blue
            else:
                color = color_dict[box_dict[ImageDetectionKey.CLASS]]

            bounding_box = BoundingBox2D(label, color, box_geometry)
            boxes.append(bounding_box)
            classes.append(box_dict[ImageDetectionKey.CLASS])
            confidences.append(box_dict[ImageDetectionKey.CONF])

        return boxes, classes, confidences


class ImageDetectorTest(ImageDetectorBase):
    """
    Sample extension of ImageDetectorBase for testing
    """
    def __init__(self, **kwargs):
        self._min_box_ratio = None
        self._max_num_detection = None
        super(ImageDetectorTest, self).__init__(**kwargs)

    def load_model(self, **kwargs):
        self._min_box_ratio = kwargs.get('min_box_ratio', 0.2)
        self._max_num_detection = kwargs.get('max_num_detection', 7)

    def _detect(self, _, orig_img_sizes):
        """ Generate random detection results based on classes and parameters in example configuration files """
        predictions = []
        for img_size in orig_img_sizes:
            boxes = []
            min_box_width = int(img_size[0] * self._min_box_ratio)
            min_box_height = int(img_size[1] * self._min_box_ratio)
            num_detection = np.random.randint(1, self._max_num_detection)
            for _ in range(1, num_detection + 1):
                # generate random class and confidence
                detected_class = self._classes[np.random.choice(self._classes.keys())]
                confidence = int(np.random.rand() * 100) / 100.

                # calculate random box
                x_min = np.random.randint(img_size[0] - min_box_width)
                y_min = np.random.randint(img_size[1] - min_box_height)

                width = np.random.randint(min_box_width, img_size[0] - x_min)
                height = np.random.randint(min_box_height, img_size[1] - y_min)

                x_max = x_min + width
                y_max = y_min + height

                # create box dictionary
                box_dict = {ImageDetectionKey.CLASS: detected_class, ImageDetectionKey.CONF: confidence,
                            ImageDetectionKey.X_MIN: x_min, ImageDetectionKey.Y_MIN: y_min,
                            ImageDetectionKey.X_MAX: x_max, ImageDetectionKey.Y_MAX: y_max}
                boxes.append(box_dict)

            predictions.append(boxes)

        return predictions


class SingleImageDetectionHandler(object):
    """
    Simple handler for ImageDetectorBase class which publishes visualized detection result for a single image message
    on a specified topic if there're subscribers. Needs to be run within a node.
    """
    _detector = None    # type: ImageDetectorBase
    _result_pub = None  # type: rospy.Publisher

    def __init__(self, detection_class, class_annotation_file, kwargs_file, result_topic):
        self._detector = detection_class(class_file=class_annotation_file, model_kwargs_file=kwargs_file)
        self._result_pub = rospy.Publisher(result_topic, ImageMsg, queue_size=1)

    def process_image_msg(self, img_msg):
        """
        Draw detected boxes and publishes if there're subscribers on self._result_pub

        :type img_msg: ImageMsg
        :return: 3-tuple:
                 - list of bounding boxes created from prediction
                 - list of detected classes
                 - list of detection confidences
        """
        predictions = self._detector.detect([img_msg])
        if len(predictions) < 1:
            raise RuntimeError('no prediction returned for image message')
        bounding_boxes, classes, confidences \
            = ImageDetectorBase.prediction_to_bounding_boxes(predictions[0], self._detector.class_colors)
        if self._result_pub.get_num_connections() > 0:
            rospy.loginfo("publishing detection result")
            drawn_img_msg = self._detector.visualize_detection(img_msg, bounding_boxes)
            self._result_pub.publish(drawn_img_msg)

        return bounding_boxes, classes, confidences

class TorchImageDetector(ImageDetectorBase):
    _model = None
    _detection_threshold = 0.
    _eval_device = None

    def __init__(self, **kwargs):
        import torch
        self._eval_device = torch.device('cuda') if torch.cuda.is_available() \
                                                 else torch.device('cpu')
        super(TorchImageDetector, self).__init__(**kwargs)

    def load_model(self, **kwargs):
        import torch

        detector_module = kwargs.get('detector_module', None)
        detector_instantiator = kwargs.get('detector_instantiator', None)
        self._detection_threshold = kwargs.get('detection_threshold', 0.)
        model_path = kwargs.get('model_path', None)

        rospy.loginfo('[load_model] Received the following model parameters:')
        rospy.loginfo('detection_module: %s', detector_module)
        rospy.loginfo('detection_instantiator: %s', detector_instantiator)
        rospy.loginfo('detection_threshold: %f', self._detection_threshold)
        rospy.loginfo('model_path: %s', model_path)
        try:
            if model_path is None:
                rospy.logwarn('[load_model] model_path not specified; loading a pretrained model')

                detector_instantiator = getattr(import_module(detector_module),
                                                detector_instantiator)
                self._model = detector_instantiator(pretrained=True)
            else:
                rospy.loginfo('[load_model] Instantiating model')
                detector_instantiator = getattr(import_module(detector_module),
                                                detector_instantiator)
                self._model = detector_instantiator(len(self._classes.keys()))

                rospy.loginfo('[load_model] Loading model parameters from %s', model_path)
                self._model.load_state_dict(torch.load(model_path, map_location=self._eval_device))
                rospy.loginfo('[load_model] Successfully loaded model')

            self._model.eval()
            self._model.to(self._eval_device)
        except TypeError as exc:
            rospy.logerr('[load_model] Error loading model')
            raise

    def _detect(self, images, orig_img_sizes):
        import torch
        from torchvision.transforms import functional

        predictions = []
        for image in images:
            # the elements of the input image have type float64, but we convert
            # them to uint8 since the evaluation is quite slow otherwise
            image = np.array(image, dtype=np.uint8)

            img_tensor = functional.to_tensor(image)
            model_predictions = None
            with torch.no_grad():
                model_predictions = self._model([img_tensor.to(self._eval_device)])
            detected_obj_data = TorchImageDetector.process_predictions(model_predictions[0],
                                                                       self._classes,
                                                                       self._detection_threshold)
            predictions.append(detected_obj_data)
        return predictions

    @staticmethod
    def process_predictions(predictions, classes, detection_threshold):
        '''Returns a list of dictionaries describing all object detections in
        "predictions". Each dictionary contains six entries:
        * the object class
        * the prediction confidence
        * four entries for the bounding box prediction described through min/max pixels over x and y

        predictions: Dict[str, Tensor] -- A dictionary containing three entries -
                                          ("boxes", "labels", and "scores") - which
                                          describe object predictions
        classes: Dict[int, str] -- A map of class labels to class names
        detection_threshold: float -- Detection threshold (between 0 and 1)

        '''
        pred_class = [classes[i] if i in classes else 'unknown'
                      for i in list(predictions['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                      for i in list(predictions['boxes'].cpu().detach().numpy())]
        pred_score = list(predictions['scores'].cpu().detach().numpy())
        pred_t = [i for i, x in enumerate(pred_score) if x > detection_threshold]

        detected_obj_data = []
        if pred_t:
            pred_t = pred_t[-1]
            pred_boxes = pred_boxes[:pred_t+1]
            pred_class = pred_class[:pred_t+1]
            pred_score = pred_score[:pred_t+1]

            num_detected_objects = len(pred_boxes)
            for i in range(num_detected_objects):
                # the results have numpy types, so we type cast them to
                # native Python types before including them in the result
                obj_data_dict = {ImageDetectionKey.CLASS: str(pred_class[i]),
                                 ImageDetectionKey.CONF: float(pred_score[i]),
                                 ImageDetectionKey.X_MIN: float(pred_boxes[i][0][0]),
                                 ImageDetectionKey.Y_MIN: float(pred_boxes[i][0][1]),
                                 ImageDetectionKey.X_MAX: float(pred_boxes[i][1][0]),
                                 ImageDetectionKey.Y_MAX: float(pred_boxes[i][1][1])}
                detected_obj_data.append(obj_data_dict)
        return detected_obj_data
