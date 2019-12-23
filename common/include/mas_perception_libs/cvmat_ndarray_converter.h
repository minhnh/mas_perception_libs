/**
 *       @file: cvmat_ndarray_converter.h
 *       @date: 2019-12-06
 *     @author: Minh Nguyen
 *  @copyright: 2019 Bonn-Rhein-Sieg University
 *    @license: LGPLv3
 */
#ifndef MAS_PERCEPTION_LIBS_CVMAT_NDARRAY_CONVERTER_H
#define MAS_PERCEPTION_LIBS_CVMAT_NDARRAY_CONVERTER_H
#include <opencv2/core/core.hpp>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace bp = boost::python;
namespace np = boost::python::numpy;

namespace mas_perception_libs
{

np::ndarray fromMatToNDArray(const cv::Mat& pMat);
cv::Mat fromNDArrayToMat(const np::ndarray& pArray);

}  // namespace mas_perception_libs

#endif  // MAS_PERCEPTION_LIBS_CVMAT_NDARRAY_CONVERTER
