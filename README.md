# MultiC
This repository contains the code for producing the results we reported in [*Applicability of Multilayer Diffusion Network Inference to Social Media Data*](https://arxiv.org/abs/2111.06235), a paper where we studied the method of inferring multilayer diffusion networks from the spreading logs of items (e.g., tweets, hashtags, news articles). Specifically, we first introduced a novel implementation of the method, *MultiC*, that achieves high efficiency and accuracy using GPU computing. Then, we conducted synthetic data experiments to explore how the inference accuracy of the method varies across a range of realistic network structure and information spreading settings. 

We provide here the code we used for generating the synthetic multilayer diffusion networks, generating the synthetic spreading logs of items, and conducting the diffusion network inference. The data generation and inference processes can be easily personalized by setting different parameter values, as we will elaborate below.

## Environmental Setup
We tested the code under Python 3.7.6, with the following libraries installed:
- [NumPy](https://numpy.org/install/)
- [SciPy](https://scipy.org/install/)
- [NetworkX](https://networkx.org/documentation/stable/install.html)
- [EoN](https://epidemicsonnetworks.readthedocs.io/en/latest/GettingStarted.html)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [PyTorch](https://pytorch.org/get-started/locally/)

In our testing, we ran the code on an NVIDIA Tesla P100 GPU card with 3,854 threads and 16GB memory. It is possible to run the code on a different GPU or a CPU, but the memory limit and the running time will vary accordingly.

## Running Instructions
In general, the code files should be run in the following order:
1. **gen_network.py**: generates the ground-truth multilayer diffusion network (saved in `/networks`)
2. **gen_logs.py**: generates the ground-truth edge transmission rates and cascade layer membership values (saved in `/truth`), and the spreading logs of the cascades (saved in `/logs`)
3. **inference_s.py**: infers the aggregated single layer diffusion network (more details about the two-phase inference in the paper; results saved in `/results`)
4. **inference_m.py**: infers the multilayer diffusion network (results saved in `/results`)

The adjustable parameters in each code file are listed at the beginning of it: simply change their values to get different data generation or inference setups. The meaning of each parameter is described briefly in the subsequent inline comment, and more detailedly in the paper. If the inline comment starts with `<W>` (resp. `<RW>`), it means the parameter value, or some other value that depends on it, will be included in the name(s) of the output (resp. input and output) file(s); therefore, after changing the value of this parameter, the results of this code snippet will be saved to a different file. Keep this in mind especially when you change the value of a parameter that is **not** included in the output file name; in such case, the new results will overwrite some earlier results. You can further change the file name structures by modifying the file I/O code snippets (search `pickle` to locate them).

## Additional Notes
1. Network overlap
2. Multiprocessing in log generation
3. Number of nonzero cascades
4. Inference on GPU/CPU
5. Removal of irrelevant edges
6. Results of single layer inference: printed/saved
7. Results of multilayer inference: printed/saved
8. Application to real data
9. Contact author if any problem/suggestion

## Citation
Please cite the following paper if you are using this code:
```
@article{xia2021applicability,
  title={Applicability of Multilayer Diffusion Network Inference to Social Media Data},
  author={Xia, Yan and Chen, Ted Hsuan Yun and Kivel{\"a}, Mikko},
  journal={arXiv preprint arXiv:2111.06235},
  year={2021}
}
```
