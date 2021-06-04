#!/usr/bin/env python
from cloud_processing_tests import PACKAGE, TEST_NAME, TestCloudProcessing


if __name__ == '__main__':
    import rosunit
    rosunit.unitrun(PACKAGE, TEST_NAME, TestCloudProcessing)
