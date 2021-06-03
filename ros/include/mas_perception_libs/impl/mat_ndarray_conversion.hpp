/**
 * @copyright 2021 Bonn-Rhein-Sieg University
 *
 * @author Minh Nguyen
 *
 * @brief Code to convert between NumPy ndarray and cv::Mat objects. More or less based on example from aFewThings gist:
 *        https://gist.github.com/aFewThings/c79e124f649ea9928bfc7bb8827f1a1c
 */
#ifndef MAS_PERCEPTION_LIBS_MAT_NDARRAY_CONVERSION_HPP
#define MAS_PERCEPTION_LIBS_MAT_NDARRAY_CONVERSION_HPP

#include <sstream>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <opencv2/opencv.hpp>

namespace bp = boost::python;
namespace bnp = boost::python::numpy;

namespace mas_perception_libs
{

bnp::ndarray matToNDArray(const cv::Mat& pMat) {
    bp::tuple shape = bp::make_tuple(pMat.rows, pMat.cols, pMat.channels());
    bp::tuple stride = bp::make_tuple(pMat.channels() * pMat.cols * sizeof(uchar),
                                      pMat.channels() * sizeof(uchar), sizeof(uchar));
    bnp::dtype dt = bnp::dtype::get_builtin<uchar>();
    bnp::ndarray ndImg = bnp::from_data(pMat.data, dt, shape, stride, bp::object());

    return ndImg;
}

cv::Mat ndArrayToMat(const bnp::ndarray& pNdArray) {
    const Py_intptr_t* shape = pNdArray.get_shape();
    char* dtype_str = bp::extract<char *>(bp::str(pNdArray.get_dtype()));

    // ndarray dimensions for creating Mat object
    int rows = shape[0];
    int cols = shape[1];
    int channel = shape[2];
    int depth;

    // for now only support 'CV_8UC3' image, which corresponds to 'uchar' type Mat. More generic solutions can be
    // added later as needed
    if (!strcmp(dtype_str, "uint8")) {
        depth = CV_8U;
    } else {
        std::ostringstream msgStream;
        msgStream << "ndArrayToMat: only support 'uint8' dtype for ndarray at the moment, received type: " << dtype_str;
        throw std::runtime_error(msgStream.str());
    }

    int type = CV_MAKETYPE(depth, channel);

    cv::Mat mat = cv::Mat(rows, cols, type);
    memcpy(mat.data, pNdArray.get_data(), sizeof(uchar) * rows * cols * channel);

    return mat;
}

}

#endif  // MAS_PERCEPTION_LIBS_MAT_NDARRAY_CONVERSION_HPP
