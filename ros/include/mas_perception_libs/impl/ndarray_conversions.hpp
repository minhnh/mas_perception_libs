/**
 * @copyright 2021 Bonn-Rhein-Sieg University
 *
 * @author Minh Nguyen
 *
 * @brief Code to convert between NumPy ndarray and other objects.
 *        - bnp::ndarray <-> cv::Mat is more or less based on example from aFewThings gist:
 *          https://gist.github.com/aFewThings/c79e124f649ea9928bfc7bb8827f1a1c
 *        - bnp::ndarray -> Eigen::Matrixxf is more or less based on example from  ndarray/Boost.NumPy GitHub repo
 *          https://github.com/ndarray/Boost.NumPy/blob/master/libs/numpy/example/wrap.cpp
 */
#ifndef MAS_PERCEPTION_LIBS_NDARRAY_CONVERSIONS_HPP
#define MAS_PERCEPTION_LIBS_NDARRAY_CONVERSIONS_HPP

#include <sstream>
#include <Eigen/Core>
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
    auto arrayDtype = pNdArray.get_dtype();

    // ndarray dimensions for creating Mat object
    int rows = shape[0];
    int cols = shape[1];
    int channel = shape[2];
    int depth;

    // for now only support 'CV_8UC3' image, which corresponds to 'uchar' type Mat. More generic solutions can be
    // added later as needed
    if (arrayDtype == bnp::dtype::get_builtin<uchar>()) {
        depth = CV_8U;
    } else {
        std::ostringstream msgStream;
        msgStream << "ndArrayToMat: only support 'uint8' dtype for ndarray at the moment, received type: "
                  << bp::extract<std::string>(bp::str(arrayDtype))();
        PyErr_SetString(PyExc_TypeError, msgStream.str().c_str());
        bp::throw_error_already_set();
    }

    int type = CV_MAKETYPE(depth, channel);

    cv::Mat mat = cv::Mat(rows, cols, type);
    memcpy(mat.data, pNdArray.get_data(), sizeof(uchar) * rows * cols * channel);

    return mat;
}

template <typename Derived>
void ndarray_to_eigenmat_2d(const bnp::ndarray &pNdArray, Eigen::MatrixBase<Derived> & pEigenMat2d) {
    std::ostringstream errMsgStream;
    if (pNdArray.get_dtype() != bnp::dtype::get_builtin<double>()) {
        errMsgStream << "ndarray_to_eigenmat_2d: only support type 'double', got: "
                     << bp::extract<std::string>(bp::str(pNdArray.get_dtype()))();
        PyErr_SetString(PyExc_TypeError, errMsgStream.str().c_str());
        bp::throw_error_already_set();
        return;
    }
    if (pNdArray.get_nd() != 2) {
        errMsgStream << "ndarray_to_eigenmat_2d: expect 2 dimensions, got: " << pNdArray.get_nd();
        PyErr_SetString(PyExc_ValueError, errMsgStream.str().c_str());
        bp::throw_error_already_set();
        return;
    }

    int rowNum = pNdArray.shape(0);
    int colNum = pNdArray.shape(1);
    if (rowNum != pEigenMat2d.rows() || colNum != pEigenMat2d.cols()) {
        PyErr_SetString(PyExc_ValueError, "ndarray_to_eigenmat_2d: eigen array dimension doesn't match source ndarray");
        bp::throw_error_already_set();
        return;
    }

    int rowStride = pNdArray.strides(0) / sizeof(double);
    int colStride = pNdArray.strides(1) / sizeof(double);
    double * arrayDataPtr = reinterpret_cast<double*>(pNdArray.get_data());
    double * rowIter = arrayDataPtr;
    double * colIter;
    for (int i = 0; i < rowNum; ++i, rowIter += rowStride) {
        colIter = rowIter;
        for (int j = 0; j < colNum; ++j, colIter += colStride) {
            pEigenMat2d(i, j) = *colIter;
        }
    }
}

}

#endif  // MAS_PERCEPTION_LIBS_NDARRAY_CONVERSIONS_HPP
