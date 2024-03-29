import os
import glob
import numpy as np
import cv2
import rospy
import rosbag
# import tf
import ros_numpy
from rospkg import RosPack
from roslib.message import get_message_class
from cv_bridge import CvBridgeError
from sensor_msgs.msg import PointCloud2, Image as ImageMsg
from mas_perception_msgs.msg import PlaneList, Object, ObjectView
from mas_perception_libs._cpp_wrapper import PlaneSegmenterWrapper, _cloud_msg_to_image_msg, \
    _crop_organized_cloud_msg, _transform_point_cloud
from .bounding_box import BoundingBox2D
from .visualization import fit_box_to_image
from .ros_message_serialization import to_cpp, from_cpp


class PlaneSegmenter(PlaneSegmenterWrapper):
    """
    Wrapper for plane segmentation code in C++
    """
    def filter_cloud(self, cloud_msg):
        """
        :type cloud_msg: PointCloud2
        :return: filtered cloud message
        :rtype: PointCloud2
        """
        serial_cloud = to_cpp(cloud_msg)
        filtered_serial_cloud = super(PlaneSegmenter, self).filter_cloud(serial_cloud)
        deserialized = from_cpp(filtered_serial_cloud, PointCloud2)
        return deserialized

    def find_planes(self, cloud_msg):
        """
        :type cloud_msg: PointCloud2
        :return: (list of detected planes, filtered cloud message)
        :rtype: (mas_perception_msgs.msg.PlaneList, sensor_msgs.msg.PointCloud2)
        """
        serial_cloud = to_cpp(cloud_msg)
        serialized_plane_list, serialized_filtered_cloud = super(PlaneSegmenter, self).find_planes(serial_cloud)
        plane_list = from_cpp(serialized_plane_list, PlaneList)
        filtered_cloud = from_cpp(serialized_filtered_cloud, PointCloud2)
        return plane_list, filtered_cloud

    def set_params(self, param_dict):
        """
        :param param_dict: parameter dictionary returned from dynarmic reconfiguration server for PlaneFitting.cfg
        :type param_dict: dict
        :return: None
        """
        super(PlaneSegmenter, self).set_params(param_dict)


def get_package_path(package_name, *paths):
    """
    return the system path of a file or folder in a ROS package
    Example: get_package_path('mas_perception_libs', 'ros', 'src', 'mas_perception_libs')

    :type package_name: str
    :param paths: chunks of the file path relative to the package
    :return: file or directory full path
    """
    rp = RosPack()
    pkg_path = rp.get_path(package_name)
    return os.path.join(pkg_path, *paths)


def get_bag_file_msg_by_type(bag_file_path, msg_type):
    """
    Extract all messages of a certain type from a bag file

    :param bag_file_path: path to ROS bag file
    :type bag_file_path: str
    :param msg_type: string name of message type, i.e. 'sensor_msgs/PointCloud2'
    :type msg_type: str
    :return: list of all messages that match the given type
    """
    bag_file = rosbag.Bag(bag_file_path)
    bag_topics = bag_file.get_type_and_topic_info()[1]
    messages = []
    for topic, msg, t in bag_file.read_messages():
        if topic not in bag_topics or bag_topics[topic].msg_type != msg_type:
            continue
        # serialize and deserialize message to get rid of bag file type
        msg_string = to_cpp(msg)
        msg = from_cpp(msg_string, get_message_class(msg_type))
        messages.append(msg)
    bag_file.close()
    return messages


def get_classes_in_data_dir(data_dir):
    """
    :type data_dir: str
    :return: dictionary mapping from indices to classes as names of top level directories
    """
    class_dict = {}
    index = 0
    for subdir in sorted(os.listdir(data_dir)):
        if os.path.isdir(os.path.join(data_dir, subdir)):
            class_dict[index] = subdir
            index += 1

    return class_dict


def process_image_message(image_msg, cv_bridge, target_size=None, func_preprocess_img=None):
    """
    perform common image processing steps on a ROS image message

    :type image_msg: ImageMsg
    :param cv_bridge: bridge for converting sensor_msgs/Image to CV image
    :param target_size: used for resizing image
    :type target_size: tuple
    :param func_preprocess_img: optional preprocess function to call on the image
    :return: processed image as CV image
    """
    np_image = None
    try:
        cv_image = cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        if target_size is not None:
            cv_image = cv2.resize(cv_image, target_size)
        np_image = np.asarray(cv_image)
        np_image = np_image.astype(float)                   # preprocess needs float64 and img is uint8
        if func_preprocess_img is not None:
            np_image = func_preprocess_img(np_image)        # Normalize the data
    except CvBridgeError as e:
        rospy.logerr('error converting to CV image: ' + str(e))
    return np_image


def case_insensitive_glob(pattern):
    """
    glob certain file types ignoring case

    :param pattern: file types (i.e. '*.jpg')
    :return: list of files matching given pattern
    """
    def either(c):
        return '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
    return glob.glob(''.join(map(either, pattern)))


def cloud_msg_to_ndarray(cloud_msg, fields=['x', 'y', 'z', 'r', 'g', 'b']):
    """
    extract data from a sensor_msgs/PointCloud2 message into a NumPy array

    :type cloud_msg: PointCloud2
    :type fields: data fields in a PointCloud2 message
    :return: NumPy array of given fields
    """
    assert isinstance(cloud_msg, PointCloud2)
    cloud_record = ros_numpy.numpify(cloud_msg)
    cloud_record = ros_numpy.point_cloud2.split_rgb_field(cloud_record)
    cloud_array = np.zeros((*cloud_record.shape, len(fields)))
    index = 0
    for field in fields:
        cloud_array[:, :, index] = cloud_record[field]
        index += 1
    return cloud_array


def crop_cloud_msg_to_ndarray(cloud_msg, bounding_box, fields=['x', 'y', 'z', 'r', 'g', 'b'], offset=0):
    """
    extract data from sensor_msgs/PointCloud2 message and crop to the dimensions in an BoundingBox2D object

    :type cloud_msg: PointCloud2
    :type bounding_box: BoundingBox2D
    :param offset: will pad bounding_box with 'offset' pixels
    :rtype: ndarray
    """
    assert isinstance(bounding_box, BoundingBox2D)

    # fit box to cloud dimensions
    bounding_box = fit_box_to_image((cloud_msg.width, cloud_msg.height), bounding_box, offset)

    cloud_array = cloud_msg_to_ndarray(cloud_msg, fields=fields)
    cloud_array = cloud_array[
        bounding_box.x: bounding_box.x + bounding_box.height,
        bounding_box.y: bounding_box.y + bounding_box.width, :]
    return cloud_array


def cloud_msg_to_cv_image(cloud_msg):
    """
    extract CV image from a sensor_msgs/PointCloud2 message

    :type cloud_msg: PointCloud2
    :return: extracted image as numpy array
    """
    return cloud_msg_to_ndarray(cloud_msg, ['r', 'g', 'b'])


def cloud_msg_to_image_msg(cloud_msg):
    """
    extract sensor_msgs/Image from a sensor_msgs/PointCloud2 message

    :type cloud_msg: PointCloud2
    :return: extracted image message
    :rtype: ImageMsg
    """
    serial_cloud = to_cpp(cloud_msg)
    serial_img_msg = _cloud_msg_to_image_msg(serial_cloud)
    return from_cpp(serial_img_msg, ImageMsg)


def crop_organized_cloud_msg(cloud_msg, bounding_box):
    """
    crop a sensor_msgs/PointCloud2 message using info from a BoundingBox2D object

    :type cloud_msg: PointCloud2
    :type bounding_box: BoundingBox2D
    :return: cropped cloud
    :rtype: PointCloud2
    """
    if not isinstance(bounding_box, BoundingBox2D):
        raise ValueError('bounding_box is not a BoundingBox2D instance')

    serial_cloud = to_cpp(cloud_msg)
    serial_cropped = _crop_organized_cloud_msg(serial_cloud, bounding_box)
    return from_cpp(serial_cropped, PointCloud2)


def crop_cloud_to_xyz(cloud_msg, bounding_box):
    """
    extract a sensor_msgs/PointCloud2 message for 3D coordinates using info from a BoundingBox2D object

    :type cloud_msg: PointCloud2
    :type bounding_box: BoundingBox2D
    :return: (x, y, z) coordinates within the bounding box
    :rtype: ndarray
    """
    return crop_cloud_msg_to_ndarray(cloud_msg, bounding_box, fields=['x', 'y', 'z'])


def transform_cloud_with_listener(cloud_msg, target_frame, tf_listener):
    try:
        common_time = tf_listener.getLatestCommonTime(target_frame, cloud_msg.header.frame_id)
        cloud_msg.header.stamp = common_time
        tf_listener.waitForTransform(target_frame, cloud_msg.header.frame_id, cloud_msg.header.stamp, rospy.Duration(1))
        tf_matrix = tf_listener.asMatrix(target_frame, cloud_msg.header)
    except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        raise RuntimeError('Unable to transform {0} -> {1}'.format(cloud_msg.header.frame_id, target_frame))

    return transform_point_cloud(cloud_msg, tf_matrix, target_frame)


def transform_point_cloud_trans_quat(cloud_msg, translation, rotation, target_frame):
    """
    transform cloud using transforms3d to calculate transformation matrix

    :type cloud_msg: PointCloud2
    :param translation: translation vector
    :type translation: list
    :param rotation: rotation quaternion, i.e. [w, x, y, z]
    :param target_frame: name of new frame to fill in cloud_msg.header.frame_id
    :return: transformed cloud
    :rtype: PointCloud2
    """
    from pytransform3d.transformations import transform_from_pq

    transform_matrix = transform_from_pq(np.append(translation, rotation))
    return transform_point_cloud(cloud_msg, transform_matrix, target_frame)


def transform_point_cloud(cloud_msg, tf_matrix, target_frame):
    """
    transform a sensor_msgs/PointCloud2 message using a transformation matrix

    :type cloud_msg: PointCloud2
    :param tf_matrix: transformation matrix
    :type tf_matrix: ndarray
    :param target_frame: frame to be transformed to
    :type target_frame: str
    :return transformed cloud
    :rtype PointCloud2
    """
    transformed_cloud = from_cpp(_transform_point_cloud(to_cpp(cloud_msg), tf_matrix), PointCloud2)
    transformed_cloud.header.frame_id = target_frame
    return transformed_cloud


def get_obj_msg_from_detection(cloud_msg, bounding_box, category, confidence, frame_id):
    '''Creates an mas_perception_msgs.msg.Object message from the given
    information about an object detected in an image generated from the
    point cloud.

    Keyword arguments:
    cloud_msg: sensor_msgs.msg.PointCloud2
    bounding_box: BoundingBox2D
    category: str
    confidence: float -- detection confidence
    frame_id: str -- frame of the resulting object

    '''
    cropped_cloud = crop_organized_cloud_msg(cloud_msg, bounding_box)
    obj_coords = crop_cloud_to_xyz(cropped_cloud, BoundingBox2D(box_geometry=(0, 0, cropped_cloud.width,
                                                                              cropped_cloud.height)))
    obj_coords = np.reshape(obj_coords, (-1, 3))
    min_coord = np.nanmin(obj_coords, axis=0)
    max_coord = np.nanmax(obj_coords, axis=0)
    mean_coord = np.nanmean(obj_coords, axis=0)

    # fill object geometry info
    detected_obj = Object()
    detected_obj.bounding_box.center.x = mean_coord[0]
    detected_obj.bounding_box.center.y = mean_coord[1]
    detected_obj.bounding_box.center.z = mean_coord[2]
    detected_obj.bounding_box.dimensions.x = max_coord[0] - min_coord[0]
    detected_obj.bounding_box.dimensions.y = max_coord[1] - min_coord[1]
    detected_obj.bounding_box.dimensions.z = max_coord[2] - min_coord[2]
    box_msg = detected_obj.bounding_box

    detected_obj.name = category
    detected_obj.category = category
    detected_obj.probability = confidence

    object_view = ObjectView()
    object_view.point_cloud = cropped_cloud
    object_view.image = cloud_msg_to_image_msg(cropped_cloud)
    detected_obj.views.append(object_view)

    detected_obj.pose.header.frame_id = frame_id
    detected_obj.pose.header.stamp = rospy.Time.now()
    detected_obj.pose.pose.position = box_msg.center
    detected_obj.pose.pose.position.x = mean_coord[0]
    detected_obj.pose.pose.orientation.w = 1.0

    detected_obj.roi.x_offset = int(bounding_box.x)
    detected_obj.roi.y_offset = int(bounding_box.y)
    detected_obj.roi.width = int(bounding_box.width)
    detected_obj.roi.height = int(bounding_box.height)

    return detected_obj
