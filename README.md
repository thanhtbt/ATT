# ATT: Adaptive Algorithms for Tensor Train Decomposition of Streaming Tensors

We propose a novel adaptive algorithm for TT decomposition of streaming tensors whose slices are serially acquired over time. By
leveraging the alternating minimization framework, our estimator minimizes an exponentially weighted least-squares cost function
in an efficient way. 


## Dependencies
+ Our code requires the [Tensor Toolbox](http://www.tensortoolbox.org/) which is already attached in this repository.
+ MATLAB R2019a


## Demo
Quick Start: Just run the file DEMO.m

## Some Results

Effect of the noise level and time-varying factors on the performance of our method

<p float="left">
  <img src="https://user-images.githubusercontent.com/26319211/110495119-7ea43b80-80f4-11eb-98a8-61256624851e.PNG" width="300" height='250' />
  <img src="https://user-images.githubusercontent.com/26319211/110495123-7f3cd200-80f4-11eb-871d-b00ff1124457.PNG" width="300" height='250' /> 
</p>
Performance of three TT decomposition algorithms in a time-varying scenario
<img src="https://user-images.githubusercontent.com/26319211/110495122-7f3cd200-80f4-11eb-8711-26da3e8ec6c4.PNG" width="300" height='250'>

## Reference: 
This code is free and open source for research purposes. If you use this code, please acknowledge the following paper.

[1] L.T. Thanh, K. Abed-Meraim, N.L. Trung, and R Boyer. "[*Adaptive Algorithms for Tensor Train Decomposition of Streaming Tensors*](https://ieeexplore.ieee.org/document/9287780)". 
**European Signal Processing Conference (EUSIPCO)**, 2020.
