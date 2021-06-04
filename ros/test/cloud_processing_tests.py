#!/usr/bin/env python
import os
import yaml
import numpy as np
import unittest
from sensor_msgs.msg import Image as ImageMsg, PointCloud2
from mas_perception_libs.bounding_box import BoundingBox2D
from mas_perception_libs.utils import get_bag_file_msg_by_type, get_package_path, cloud_msg_to_cv_image, \
    cloud_msg_to_image_msg, crop_cloud_to_xyz, PlaneSegmenter, crop_organized_cloud_msg, \
    transform_point_cloud_trans_quat
from mas_perception_libs.visualization import draw_labeled_boxes, crop_image


PACKAGE = 'mas_perception_libs'
TEST_NAME = 'cloud_processing'


def get_random_boxes_and_offsets(img_width, img_height, box_num):
    boxes = []
    for i in range(box_num):
        y = np.random.randint(0, img_height - 1)
        x = np.random.randint(0, img_width - 1)
        if i < box_num / 2:
            # normal case
            width = np.random.randint(0, img_width - x)
            height = np.random.randint(0, img_height - y)
        else:
            # box has region outside of frame dimension
            width = np.random.randint(img_width - x, img_width)
            height = np.random.randint(img_height - y, img_height)

        box = BoundingBox2D(box_geometry=(x, y, width, height))
        boxes.append(box)

    # half of boxes will come with non-zero random offsets
    offsets = np.zeros(box_num)
    for i in range(int(box_num / 2)):
        offsets[i] = np.random.randint(1, np.min([img_width, img_height]) / 4)
    np.random.shuffle(offsets)

    return boxes, offsets


class TestCloudProcessing(unittest.TestCase):

    _cloud_messages = None
    _crop_boxes_num = None
    _transform_num = None
    _plane_segmenter_configs = None

    @classmethod
    def setUpClass(cls):
        config_file = os.path.join('config', TEST_NAME + '.yaml')
        cls.assertTrue(os.path.exists(config_file), 'test configuration file does not exist: ' + config_file)

        with open(config_file, 'r') as infile:
            configs = yaml.load(infile, Loader=yaml.SafeLoader)

        # load point clouds from bag file
        bag_file_config = configs['bag_file']
        bag_file_path = get_package_path(bag_file_config['package'], *bag_file_config['path'])
        cls.assertTrue(os.path.exists(bag_file_path),
                       'bag file containing point clouds does not exist: ' + bag_file_path)
        cls._cloud_messages = get_bag_file_msg_by_type(bag_file_path, 'sensor_msgs/PointCloud2')
        cls.assertTrue(cls._cloud_messages and len(cls._cloud_messages) > 0,
                       "no 'sensor_msgs/PointCloud2' message in bag file: " + bag_file_path)

        # load configurations for cloud utility functions
        cls._crop_boxes_num = configs['crop_boxes_num']
        cls._transform_num = configs['transform_num']

        # load configurations for plane segmentation
        cls._plane_segmenter_configs = configs['plane_segmenter']

        # seed np.random once
        np.random.seed(1234)

    def test_cloud_msg_to_image_conversion(self):
        for cloud_msg in TestCloudProcessing._cloud_messages:
            cv_image = cloud_msg_to_cv_image(cloud_msg)
            self.assertIsInstance(cv_image, np.ndarray,
                                  "'cloud_msg_to_cv_image' does not return type 'numpy.ndarray'")

            # test draw_label_boxes()
            boxes, offsets = get_random_boxes_and_offsets(cloud_msg.width, cloud_msg.height,
                                                          TestCloudProcessing._crop_boxes_num)

            drawn_img = draw_labeled_boxes(cv_image.astype(np.uint8), boxes)
            self.assertIsInstance(drawn_img, np.ndarray, "'draw_labeled_boxes' does not return type 'numpy.ndarray'")

            # test crop_image()
            for index, box in enumerate(boxes):
                cropped = crop_image(drawn_img, box, int(offsets[index]))
                self.assertIsInstance(cropped, np.ndarray)

            # test for expected box out of bound exception
            box = BoundingBox2D(box_geometry=(cloud_msg.width + offsets[0] - 1, cloud_msg.height + offsets[0] - 1,
                                              boxes[0].width, boxes[0].height))
            with self.assertRaises(RuntimeError,
                                   msg="crop_image failed to raise exception when box coords are too large"):
                crop_image(drawn_img, box, int(offsets[0]))

            # test cloud_msg_to_image_msg()
            image_msg = cloud_msg_to_image_msg(cloud_msg)
            self.assertIsInstance(image_msg, ImageMsg,
                                  "'cloud_msg_to_image_msg' does not return type 'sensor_msgs/Image'")

    def test_cloud_cropping(self):
        for cloud_msg in TestCloudProcessing._cloud_messages:
            cloud_width = cloud_msg.width
            cloud_height = cloud_msg.height
            boxes, _ = get_random_boxes_and_offsets(cloud_width, cloud_height, TestCloudProcessing._crop_boxes_num)
            for box in boxes:
                cropped_cloud = crop_organized_cloud_msg(cloud_msg, box)
                self.assertIs(type(cropped_cloud), PointCloud2,
                              "'crop_organized_cloud_msg' does not return type 'sensor_msgs/PointCloud2'")
                self.assertTrue(box.x + cropped_cloud.width <= cloud_width and
                                box.y + cropped_cloud.height <= cloud_height,
                                "'crop_organized_cloud_msg' does not handle large box dimensions correctly")

                cropped_xyz = crop_cloud_to_xyz(cloud_msg, box)
                self.assertTrue(type(cropped_xyz) == np.ndarray and len(cropped_xyz.shape) == 3)
                self.assertTrue(box.y + cropped_xyz.shape[0] <= cloud_height and
                                box.x + cropped_xyz.shape[1] <= cloud_width,
                                "'crop_cloud_to_xyz' does not handle large box dimensions correctly")

    def test_transform_point_cloud_trans_quat(self):
        from pytransform3d.rotations import random_quaternion

        frame_name = 'test_frame'
        for cloud_msg in TestCloudProcessing._cloud_messages:
            for _ in range(TestCloudProcessing._transform_num):
                translation = np.random.rand(3) * 1.0       # scale to 1 meter
                rotation = random_quaternion()              # quaternion
                transformed_cloud = transform_point_cloud_trans_quat(cloud_msg, translation, rotation, frame_name)
                self.assertIs(type(transformed_cloud), PointCloud2,
                              "'transform_point_cloud_trans_quat' does not return type 'sensor_msgs/PointCloud2'")
                self.assertTrue(transformed_cloud.header.frame_id == frame_name)

    def test_plane_segmenter(self):
        plane_segmenter = PlaneSegmenter()

        # set plane segmentation configurations
        config_file_info = TestCloudProcessing._plane_segmenter_configs['config_file']
        config_file_path = get_package_path(config_file_info['package'], *config_file_info['path'])
        self.assertTrue(os.path.exists(config_file_path),
                        'config file for PlaneSegmenter does not exist: ' + config_file_path)
        with open(config_file_path) as infile:
            configs = yaml.load(infile, Loader=yaml.SafeLoader)
        plane_segmenter.set_params(configs)

        # extract transformation info for point clouds
        transform_info = TestCloudProcessing._plane_segmenter_configs['target_frame']
        translation = transform_info['translation']
        rotation = transform_info['rotation']
        frame_name = transform_info['name']
        for cloud_msg in TestCloudProcessing._cloud_messages:
            transformed_cloud = transform_point_cloud_trans_quat(cloud_msg, translation, rotation, frame_name)
            # TODO (minhnh) test if filtering does what it's supposed to do, test edge cases
            _ = plane_segmenter.filter_cloud(transformed_cloud)
            # TODO(minhnh) test multiple plane segmentation, test if returned cloud is filtered
            plane_list, _ = plane_segmenter.find_planes(transformed_cloud)
            self.assertTrue(len(plane_list.planes) > 0, 'plane segmentation did not detect any plane')


if __name__ == '__main__':
    unittest.main()
