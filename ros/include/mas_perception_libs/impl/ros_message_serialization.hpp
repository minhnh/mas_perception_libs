#ifndef MAS_PERCEPTION_LIBS_ROS_MESSAGE_SERIALIZATION_H
#define MAS_PERCEPTION_LIBS_ROS_MESSAGE_SERIALIZATION_H

#include <string>
#include <Python.h>
#include <boost/python.hpp>
#include <boost/shared_array.hpp>
#include <ros/serialization.h>

namespace bp = boost::python;

namespace mas_perception_libs
{
    template<typename M>
    M from_python(const bp::object & pPyBuffer)
    {
        // ensure that object is a Python buffer
        PyObject* bufferPtr = pPyBuffer.ptr();
        if (!PyObject_CheckBuffer(bufferPtr)) {
            PyErr_SetString(PyExc_TypeError, "data not of buffer type");
            bp::throw_error_already_set();
        }

        // create view into buffer object, copy data, then release view
        Py_buffer view;
        PyObject_GetBuffer(bufferPtr, &view, PyBUF_SIMPLE);
        auto pyBufPtr = (uint8_t *) view.buf;
        auto pyBufLen = (int) view.len;
        boost::shared_array<uint8_t> buffer(new uint8_t[pyBufLen]);
        for (size_t i = 0; i < pyBufLen; ++i)
        {
            buffer[i] = pyBufPtr[i];
        }
        PyBuffer_Release(&view);

        // deserialize message
        ros::serialization::IStream stream(buffer.get(), pyBufLen);
        M msg;
        try {
            ros::serialization::Serializer<M>::read(stream, msg);
        } catch (const std::exception &exp) {
            std::string errMsg("failed to deserialize message: ");
            errMsg.append(exp.what());
            PyErr_SetString(PyExc_RuntimeError, errMsg.c_str());
            bp::throw_error_already_set();
        }
        return msg;
    };

    template<typename M>
    bp::object to_python(const M& pRosMsg)
    {
        // create new char buffer and serialize the ROS message into it
        size_t serialSize = ros::serialization::serializationLength(pRosMsg);
        char * buffer = new char[serialSize];
        ros::serialization::OStream stream((uint8_t *) buffer, static_cast<uint32_t>(serialSize));
        ros::serialization::serialize(stream, pRosMsg);

        // create python memory view object and register it with Boost
        PyObject* memView = PyMemoryView_FromMemory(buffer, serialSize, PyBUF_WRITE);
        return bp::object(bp::handle<>(memView));
    };
}

#endif //MAS_PERCEPTION_LIBS_ROS_MESSAGE_SERIALIZATION_H
