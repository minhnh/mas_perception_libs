/*!
 * @copyright 2018 Bonn-Rhein-Sieg University
 *
 * @author Minh Nguyen
 *
 * @brief File contains C++ definitions that are made available in Python using the Boost Python library.
 *        Detailed descriptions of parameters are in the Python source files
 */
#include <mas_perception_libs/use_numpy.h>
#include <mas_perception_libs/impl/ros_message_serialization.hpp>
#include <mas_perception_libs/bounding_box_wrapper.h>
#include <mas_perception_libs/image_bounding_box.h>
#include <mas_perception_libs/bounding_box_2d.h>
#include <mas_perception_libs/point_cloud_utils_ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/core/eigen.hpp>
#include <pcl_ros/transforms.h>
#include <boost/python.hpp>
#include <Eigen/Core>
#include <vector>
#include <string>

namespace bp = boost::python;
using BoundingBox = mas_perception_libs::BoundingBox;

namespace mas_perception_libs
{

class PlaneSegmenterWrapper : PlaneSegmenterROS
{
public:
    /*!
     * @brief wrapper for exposing in Python a function to set PlaneFitting dynamic reconfiguration values for the
     *        SacPlaneSegmenter and CloudFilter objects
     */
    void
    setParams(const bp::dict & pConfigDict)
    {
        CloudFilterParams cfParams;
        if (!pConfigDict.contains("passthrough_limit_min_x"))
            throw std::invalid_argument("Python config dict does not contain key 'passthrough_limit_min_x'");
        cfParams.mPassThroughLimitMinX = bp::extract<float>(pConfigDict["passthrough_limit_min_x"]);

        if (!pConfigDict.contains("passthrough_limit_max_x"))
            throw std::invalid_argument("Python config dict does not contain key 'passthrough_limit_max_x'");
        cfParams.mPassThroughLimitMaxX = bp::extract<float>(pConfigDict["passthrough_limit_max_x"]);

        if (!pConfigDict.contains("passthrough_limit_min_y"))
            throw std::invalid_argument("Python config dict does not contain key 'passthrough_limit_min_y'");
        cfParams.mPassThroughLimitMinY = bp::extract<float>(pConfigDict["passthrough_limit_min_y"]);

        if (!pConfigDict.contains("passthrough_limit_max_y"))
            throw std::invalid_argument("Python config dict does not contain key 'passthrough_limit_max_y'");
        cfParams.mPassThroughLimitMaxY = bp::extract<float>(pConfigDict["passthrough_limit_max_y"]);

        if (!pConfigDict.contains("voxel_limit_min_z"))
            throw std::invalid_argument("Python config dict does not contain key 'voxel_limit_min_z'");
        cfParams.mVoxelLimitMinZ = bp::extract<float>(pConfigDict["voxel_limit_min_z"]);

        if (!pConfigDict.contains("voxel_limit_max_z"))
            throw std::invalid_argument("Python config dict does not contain key 'voxel_limit_max_z'");
        cfParams.mVoxelLimitMaxZ = bp::extract<float>(pConfigDict["voxel_limit_max_z"]);

        if (!pConfigDict.contains("voxel_leaf_size"))
            throw std::invalid_argument("Python config dict does not contain key 'voxel_leaf_size'");
        cfParams.mVoxelLeafSize = bp::extract<float>(pConfigDict["voxel_leaf_size"]);

        mCloudFilter.setParams(cfParams);

        SacPlaneSegmenterParams planeFitParams;
        if (!pConfigDict.contains("normal_radius_search"))
            throw std::invalid_argument("Python config dict does not contain key 'normal_radius_search'");
        planeFitParams.mNormalRadiusSearch = bp::extract<double>(pConfigDict["normal_radius_search"]);

        if (!pConfigDict.contains("sac_max_iterations"))
            throw std::invalid_argument("Python config dict does not contain key 'sac_max_iterations'");
        planeFitParams.mSacMaxIterations = bp::extract<int>(pConfigDict["sac_max_iterations"]);

        if (!pConfigDict.contains("sac_distance_threshold"))
            throw std::invalid_argument("Python config dict does not contain key 'sac_distance_threshold'");
        planeFitParams.mSacDistThreshold = bp::extract<double>(pConfigDict["sac_distance_threshold"]);

        if (!pConfigDict.contains("sac_optimize_coefficients"))
            throw std::invalid_argument("Python config dict does not contain key 'sac_optimize_coefficients'");
        planeFitParams.mSacOptimizeCoeffs = bp::extract<bool>(pConfigDict["sac_optimize_coefficients"]);

        if (!pConfigDict.contains("sac_eps_angle"))
            throw std::invalid_argument("Python config dict does not contain key 'sac_eps_angle'");
        planeFitParams.mSacEpsAngle = bp::extract<double>(pConfigDict["sac_eps_angle"]);

        if (!pConfigDict.contains("sac_normal_distance_weight"))
            throw std::invalid_argument("Python config dict does not contain key 'sac_normal_distance_weight'");
        planeFitParams.mSacNormalDistWeight = bp::extract<double>(pConfigDict["sac_normal_distance_weight"]);

        mPlaneSegmenter.setParams(planeFitParams);
    }

    /*!
     * @brief wrapper to expose in Python a function to filter point clouds using the passthrough and voxel filters
     */
    bp::object
    filterCloud(const bp::object & pSerialCloud)
    {
        auto cloudMsg = from_python<sensor_msgs::PointCloud2>(pSerialCloud);
        sensor_msgs::PointCloud2::Ptr cloudMsgPtr = boost::make_shared<sensor_msgs::PointCloud2>(cloudMsg);
        auto filteredCloudPtr = PlaneSegmenterROS::filterCloud(cloudMsgPtr);
        return to_python(*filteredCloudPtr);
    }

    /*!
     * @brief wrapper to expose in Python a function to fit plane(s) from point clouds
     */
    bp::tuple
    findPlanes(const bp::object & pSerialCloud)
    {
        auto cloudMsg = from_python<sensor_msgs::PointCloud2>(pSerialCloud);
        sensor_msgs::PointCloud2::Ptr cloudMsgPtr = boost::make_shared<sensor_msgs::PointCloud2>(cloudMsg);
        // also fit plane
        auto filteredCloudPtr = boost::make_shared<sensor_msgs::PointCloud2>();
        mas_perception_msgs::PlaneList::Ptr planeListPtr;
        try
        {
            planeListPtr = PlaneSegmenterROS::findPlanes(cloudMsgPtr, filteredCloudPtr);
        }
        catch (std::runtime_error &ex)
        {
            // TODO(minhnh) make actual Python exceptions
            throw;
        }
        auto serializedFilteredCloud = to_python(*filteredCloudPtr);
        auto serializedPlanes = to_python(*planeListPtr);
        return bp::make_tuple<bp::object, bp::object>(serializedPlanes, serializedFilteredCloud);
    }
};

struct BoundingBox2DWrapper : BoundingBox2D
{
    // TODO(minhnh): expose color
    /*!
     * @brief Constructor for the extension of BoundingBox2D for use in Python
     */
    BoundingBox2DWrapper(const std::string &pLabel, const bp::tuple &pColor, const bp::tuple &pBox) : BoundingBox2D()
    {
        mLabel = pLabel;

        // extract color
        if (bp::len(pColor) != 3)
        {
            throw std::invalid_argument("pColor is not a 3-tuple containing integers");
        }
        mColor = CV_RGB(static_cast<int>(bp::extract<double>(pColor[0])),
                        static_cast<int>(bp::extract<double>(pColor[1])),
                        static_cast<int>(bp::extract<double>(pColor[2])));

        // extract box geometry
        if (bp::len(pBox) != 4)
        {
            throw std::invalid_argument("box geometry is not a tuple containing 4 numerics");
        }
        mX = static_cast<int>(bp::extract<double>(pBox[0]));
        mY = static_cast<int>(bp::extract<double>(pBox[1]));
        mWidth = static_cast<int>(bp::extract<double>(pBox[2]));
        mHeight = static_cast<int>(bp::extract<double>(pBox[3]));
    }
};

/*!
 * @brief crops object images from a ROS image messages using ImageBoundingBox. Legacy from mcr_scene_segmentation.
 */
bp::tuple
getCropsAndBoundingBoxes(
    const bp::object & pSerialImageMsg, const bp::object & pSerialCameraInfo, const bp::object & pSerialBoundingBoxList
) {
    const sensor_msgs::Image &imageMsg = from_python<sensor_msgs::Image>(pSerialImageMsg);
    const sensor_msgs::CameraInfo &camInfo = from_python<sensor_msgs::CameraInfo>(pSerialCameraInfo);
    const mas_perception_msgs::BoundingBoxList &boundingBoxList
            = from_python<mas_perception_msgs::BoundingBoxList>(pSerialBoundingBoxList);
    ImageBoundingBox mImgBoundingBox(imageMsg, camInfo, boundingBoxList);

    // serialize image list
    const mas_perception_msgs::ImageList &imageList = mImgBoundingBox.cropped_image_list();
    auto serialImageList = to_python(imageList);

    // convert vector to bp::list
    const std::vector<std::vector<cv::Point2f>> &boxVerticesVect = mImgBoundingBox.box_vertices_vector();
    bp::list boxVerticesList;
    for (const auto &boxVertices : boxVerticesVect)
    {
        bp::list boostBoxVertices;
        for (const auto &vertex : boxVertices)
        {
            boost::array<float, 2> boostVertex{};
            boostVertex[0] = vertex.x;
            boostVertex[1] = vertex.y;
            boostBoxVertices.append(boostVertex);
        }
        boxVerticesList.append(boostBoxVertices);
    }

    // return result tuple
    return bp::make_tuple<bp::object, bp::list>(serialImageList, boxVerticesList);
}

/*!
 * @brief Adjust BoundingBox2D geometry to fit within an image, wrapper for C++ function fitBoxToImage
 */
BoundingBox2DWrapper
fitBoxToImageWrapper(const bp::tuple &pImageSizeTuple, BoundingBox2DWrapper pBox, int pOffset)
{
    if (bp::len(pImageSizeTuple) != 2)
    {
        throw std::invalid_argument("image size is not a tuple containing 2 numerics");
    }
    int width = static_cast<int>(bp::extract<double>(pImageSizeTuple[0]));
    int height = static_cast<int>(bp::extract<double>(pImageSizeTuple[1]));
    cv::Size imageSize(width, height);
    cv::Rect adjustedBox = fitBoxToImage(imageSize, pBox.getCvRect(), pOffset);
    pBox.updateBox(adjustedBox);
    return pBox;
}

/*!
 * @brief Python wrapper for the PCL conversion function toROSMsg which converts a sensor_msgs/PointCloud2 object to a
 *        sensor_msgs/Image object
 */
bp::object
cloudMsgToImageMsgWrapper(const bp::object & pSerialCloud)
{
    // unserialize cloud message
    sensor_msgs::PointCloud2 cloudMsg = from_python<sensor_msgs::PointCloud2>(pSerialCloud);

    // check for organized cloud and extract image message
    if (cloudMsg.height <= 1)
    {
        throw std::invalid_argument("Input point cloud is not organized!");
    }
    sensor_msgs::Image imageMsg;
    pcl::toROSMsg(cloudMsg, imageMsg);

    return to_python(imageMsg);
}

/*!
 * @brief Crop a sensor_msgs/PointCloud2 message using a BoundingBox2D object, wrapper for C++ function
 *        cropOrganizedCloudMsg
 */
bp::object
cropOrganizedCloudMsgWrapper(const bp::object & pSerialCloud, BoundingBox2DWrapper pBox)
{
    // unserialize cloud message
    sensor_msgs::PointCloud2 cloudMsg = from_python<sensor_msgs::PointCloud2>(pSerialCloud);

    sensor_msgs::PointCloud2 croppedCloudMsg;
    cropOrganizedCloudMsg(cloudMsg, pBox, croppedCloudMsg);

    return to_python(croppedCloudMsg);
}

/* TODO(minhnh) expose Color and other optional params */
bp::object
planeMsgToMarkerWrapper(const bp::object & pSerialPlane, const std::string &pNamespace)
{
    auto plane = from_python<mas_perception_msgs::Plane>(pSerialPlane);
    auto markerPtr = planeMsgToMarkers(plane, pNamespace);
    return to_python(*markerPtr);
}

}  // namespace mas_perception_libs

BOOST_PYTHON_MODULE(_cpp_wrapper)
{
    using mas_perception_libs::BoundingBoxWrapper;
    using mas_perception_libs::PlaneSegmenterWrapper;
    using mas_perception_libs::BoundingBox2DWrapper;

    bp::class_<BoundingBoxWrapper>("BoundingBoxWrapper", bp::init<const bp::object &, bp::list&>())
            .def("get_pose", &BoundingBoxWrapper::getPose)
            .def("get_ros_message", &BoundingBoxWrapper::getRosMsg);

    bp::class_<PlaneSegmenterWrapper>("PlaneSegmenterWrapper")
            .def("set_params", &PlaneSegmenterWrapper::setParams)
            .def("filter_cloud", &PlaneSegmenterWrapper::filterCloud)
            .def("find_planes", &PlaneSegmenterWrapper::findPlanes);

    bp::def("get_crops_and_bounding_boxes_wrapper", mas_perception_libs::getCropsAndBoundingBoxes);

    bp::class_<BoundingBox2DWrapper>("BoundingBox2DWrapper", bp::init<std::string, bp::tuple&, bp::tuple&>())
            .def_readwrite("x", &BoundingBox2DWrapper::mX)
            .def_readwrite("y", &BoundingBox2DWrapper::mY)
            .def_readwrite("width", &BoundingBox2DWrapper::mWidth)
            .def_readwrite("height", &BoundingBox2DWrapper::mHeight)
            .def_readwrite("label", &BoundingBox2DWrapper::mLabel);

    bp::def("_fit_box_to_image", mas_perception_libs::fitBoxToImageWrapper);

    bp::def("_cloud_msg_to_image_msg", mas_perception_libs::cloudMsgToImageMsgWrapper);

    bp::def("_plane_msg_to_marker", mas_perception_libs::planeMsgToMarkerWrapper);

    bp::def("_crop_organized_cloud_msg", mas_perception_libs::cropOrganizedCloudMsgWrapper);
}
