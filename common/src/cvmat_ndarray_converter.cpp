/**
 *       @file: cvmat_ndarray_converter.cpp
 *       @date: 2019-12-06
 *     @author: Minh Nguyen
 *  @copyright: 2019 Bonn-Rhein-Sieg University
 *    @license: LGPLv3
 */
#include <mas_perception_libs/cvmat_ndarray_converter.h>

namespace mas_perception_libs
{

np::ndarray fromMatToNDArray(const cv::Mat& pMat) {
    bp::tuple shape = bp::make_tuple(pMat.rows, pMat.cols, pMat.channels());
    bp::tuple stride = bp::make_tuple(pMat.channels() * pMat.cols * sizeof(uchar), pMat.channels() * sizeof(uchar), sizeof(uchar));
    np::dtype dt = np::dtype::get_builtin<uchar>();
    np::ndarray ndImg = np::from_data(pMat.data, dt, shape, stride, bp::object());

    return ndImg;
}

cv::Mat fromNDArrayToMat(const np::ndarray& pArray) {
    //int length = pArray.get_nd(); // get_nd() returns num of dimensions. this is used as a length, but we don't need to use in this case. because we know that image has 3 dimensions.
    const Py_intptr_t* shape = pArray.get_shape(); // get_shape() returns Py_intptr_t* which we can get the size of n-th dimension of the ndarray.
    char* dtype_str = bp::extract<char *>(bp::str(pArray.get_dtype()));

    // variables for creating Mat object
    int rows = shape[0];
    int cols = shape[1];
    int channel = shape[2];
    int depth;

    // you should find proper type for c++. in this case we use 'CV_8UC3' image, so we need to create 'uchar' type Mat.
    if (!strcmp(dtype_str, "uint8")) {
        depth = CV_8U;
    }
    else {
        throw std::runtime_error("input cloud is not organized");
    }

    int type = CV_MAKETYPE(depth, channel); // CV_8UC3

    cv::Mat mat = cv::Mat(rows, cols, type);
    memcpy(mat.data, pArray.get_data(), sizeof(uchar) * rows * cols * channel);

    return mat;
}

}  // namespace mas_perception_libs
