#!/usr/bin/env python

import unittest
from clens.lensing import cov_gammat
from clens.lensing import cov_DeltaSigma

class TestLensing(unittest.TestCase):

    # def test_cov_gammat(self):
    #     cov_gammat.demo_cov(save_files=True)

    def test_DeltaSigma(self):
        cov_DeltaSigma.demo_cov()



if __name__ == '__main__':
    unittest.main()