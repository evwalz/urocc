<!-- README.md is generated from README.Rmd. Please edit that file -->
urocc
====

<!-- badges: start -->
<!-- badges: end -->
The urocc package provides the functionality of creating an animated ROC movie (ROCM), a universal ROC (UROC) curve and to compute the coefficient of predictive ability (CPA). These tools generalize the classical ROC curve and AUC and can be applied to assess the predictive abilities of features, markers and tests for not only binary classification problems but for just any ordinal or real-valued outcome.
<br/>
For more information see: https://arxiv.org/abs/1912.01956

Installation
------------

To build and install clone repository and run:

``` r
pip install ./urocc
```

Example
-------

The following basic example shows how to create a UROC curve:

``` r
from urocc import uroc

response = [1,2,3,4,5,2,4,3,5,6,7]
predictor = [3,5,4,6,7,8,9,10,9,12,13]

uroc_curve = uroc(response, forecast)
uroc_curve.plot()
```



