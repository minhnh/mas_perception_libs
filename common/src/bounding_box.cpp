/*
 * Copyright 2016 Bonn-Rhein-Sieg University
 *
 * Author: Sergey Alexandrov
 *
 */

#include <pcl/common/transforms.h>
#include <opencv2/opencv.hpp>

#include <mas_perception_libs/bounding_box.h>
#include <limits>
#include <algorithm>

namespace mas_perception_libs
{

BoundingBox BoundingBox::create(const PointCloud::ConstPtr& cloud,
                                const Eigen::Vector3f& normal)
{
    BoundingBox box;

    // Step 1: transform the cloud so that z-axis as aligned with plane normal.
    Eigen::Vector3f perpendicular(-normal[1], normal[0], normal[2]);
    Eigen::Affine3f transform = pcl::getTransFromUnitVectorsZY(normal, perpendicular);
    Eigen::Affine3f inverse_transform = transform.inverse(Eigen::Isometry);
    PointCloud cloud_transformed;
    pcl::transformPointCloud(*cloud, cloud_transformed, transform);

    std::vector<cv::Point> points;

    // CvPoints are made of integers, so we will need to scale our points (which are in meters).
    const float SCALE = 1000.0;
    float min_z = std::numeric_limits<float>::max();
    float max_z = -1 * std::numeric_limits<float>::max();

    for (const auto &pt : cloud_transformed.points)
    {
        if (!std::isnan(pt.z))
        {
            cv::Point p;
            p.x = static_cast<int>(pt.x * SCALE);
            p.y = static_cast<int>(pt.y * SCALE);
            points.push_back(p);
            if (pt.z > max_z)
                max_z = pt.z;
            if (pt.z < min_z)
                min_z = pt.z;
        }
    }

    cv::RotatedRect box2d = cv::minAreaRect(points);
    box.dimensions_[0] = max_z - min_z;
    box.dimensions_[1] = std::max(box2d.size.width, box2d.size.height) / SCALE;
    box.dimensions_[2] = std::min(box2d.size.width, box2d.size.height) / SCALE;
    box.center_[0] = box2d.center.x / SCALE;
    box.center_[1] = box2d.center.y / SCALE;
    box.center_[2] = min_z + box.dimensions_[0] / 2.0;
    box.center_ = inverse_transform * box.center_;

    cv::Point2f vertices[4];
    cv::RotatedRect(box2d).points(vertices);
    for (const auto &pt : vertices)
    {
        Eigen::Vector3f p;
        p << pt.x / SCALE, pt.y / SCALE, min_z;
        box.vertices_.push_back(inverse_transform * p);
    }
    for (const auto &pt : vertices)
    {
        Eigen::Vector3f p;
        p << pt.x / SCALE, pt.y / SCALE, max_z;
        box.vertices_.push_back(inverse_transform * p);
    }
    return box;
}

BoundingBox BoundingBox::create(const PointCloud::VectorType& points,
                                const Eigen::Vector3f& normal)
{
    PointCloud::Ptr cloud(new PointCloud);
    cloud->points = points;
    cloud->width = static_cast<uint32_t>(points.size());
    cloud->height = 1;
    return create(cloud, normal);
}

}  // namespace mas_perception_libs
