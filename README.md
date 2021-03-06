# MultiC
This repository contains the code for producing the results we reported in [*Limits of Multilayer Diffusion Network Inference in Social Media Research*](https://ojs.aaai.org/index.php/ICWSM/article/view/19365), a paper where we studied the method of inferring multilayer diffusion networks from the spreading logs of items (e.g., tweets, hashtags, news articles). In the paper, we first introduced a novel implementation of the method, *MultiC*, that achieves high efficiency and accuracy using GPU computing. Then, we conducted synthetic data experiments to explore how the inference accuracy of the method varies across a range of realistic network structure and information spreading settings. 

We provide here the code we used for i) generating the synthetic multilayer diffusion networks, ii) generating the synthetic spreading logs of items, and iii) conducting the diffusion network inference. The data generation and inference processes can be easily personalized by setting different parameter values, as we will elaborate below.

## Environmental Setup
We tested the code under Python 3.7.6, with the following libraries installed:
- [`NumPy`](https://numpy.org/install/)
- [`SciPy`](https://scipy.org/install/)
- [`NetworkX`](https://networkx.org/documentation/stable/install.html)
- [`EoN`](https://epidemicsonnetworks.readthedocs.io/en/latest/GettingStarted.html)
- [`scikit-learn`](https://scikit-learn.org/stable/install.html)
- [`PyTorch`](https://pytorch.org/get-started/locally/)

In our testing, we ran the code on an NVIDIA Tesla P100 GPU card with 3,854 threads and 16GB memory. It is possible to run the code on a different GPU or a CPU, but the memory limit and the running time will vary accordingly.

## Running the Code
In general, the code files should be run in the following order:
1. `gen_network.py`: generates the ground-truth multilayer diffusion network (saved in `/networks`)
2. `gen_logs.py`: generates the ground-truth edge transmission rates and cascade layer membership values (saved in `/truth`), and the spreading logs of the cascades (saved in `/logs`)
3. `inference_s.py`: infers the aggregated single layer diffusion network (more details about the two-phase inference in the paper; results saved in `/results`)
4. `inference_m.py`: infers the multilayer diffusion network (results saved in `/results`)

The adjustable parameters in each code file are listed at the beginning of it: simply change their values to get different data generation or inference setups. The meaning of each parameter is described briefly in the subsequent inline comment, and more detailedly in the paper. If the inline comment starts with `<W>` (resp. `<RW>`), it means the parameter value, or some other value that depends on it, will be included in the name(s) of the output (resp. input and output) file(s); therefore, after changing the value of this parameter, the results of this code snippet will be saved to a different file. Keep this in mind especially when you change the value of a parameter that is **not** included in the output file name; in such case, the new results will overwrite some earlier results. You can further change the file name structures by modifying the file I/O code snippets (search `pickle` to locate them).

## Evaluating the Results
#### Single layer phase
During the single layer phase (i.e., when executing `inference_s.py`) that infers edge existence in the aggregated single layer network, the code prints out after each iteration the value of the objective function (i.e., the negative log likelihood of data), as well as the [ROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) [AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html) score and the [PRC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) AUC score of the inferred edges in the aggregated network. If the inference works properly, the objective value should decrease and the AUC scores should increase. After the optimization ends, the code prints out the running time of respectively data parsing, tensor construnction, inference, and result evaluation. When a GPU is used, the code also prints out the maximum GPU memory usage throughout the inference process. The statistics described above as well as the set of inferred edges are then saved as a Python dictionary (check the final part of the code for its exact structure) to a file under `/results` with prefix `s_`.

#### Multilayer phase
During the multilayer phase (i.e., when executing `inference_m.py`) that infers the layer-wise transmission rates and the layer membership of cascades, the code prints out after each iteration the value of the objective function (i.e., the negative log likelihood of data), as well as the classification accuracy of the cascade layer membership variables (i.e., `pi`). After the optimization ends, the code prints out the accuracy of the inferred transmission rates (i.e., `alpha`), evaluated by respectively [Spearman's rank-order correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html), ROC AUC, and PRC AUC. As in the single layer phase, the code prints out the running time of the code, as well as the maximum GPU memory usage if a GPU is used. The statistics are then saved as a Python dictionary (check the final part of the code for its exact structure) to a file under `/results` with prefix `m_`.

## Reference Statistics
Under the setting in our provided code (also the core setting in our paper), a network of 1000 nodes, 2 layers, and 4422 edges (aggregated) is generated. Using 20 CPU cores on an Intel Xeon Gold 6248 @ 2.50 GHz Processor, 442200 cascades are generated in roughly 1 hour and 13 minutes. (Note that it is possible to generate the cascades much faster when there is no layer mixing in cascade spreading, but we implement it in the current way to support cases where there is layer mixing.)

On an NVIDIA Tesla P100 GPU card with 3,854 threads and 16GB memory, the single layer inference finishes 500 iterations in roughly 50 minutes, with maximum 0.5 GB GPU memory usage, an end ROC AUC score of 0.9974, and an end PRC AUC score of 0.9089. (Note that the optimization already reaches decent convergence after around 100 iterations, so you can cut the runtime by decreasing `min_iter` and `max_iter`. You can also cut the runtime by increasing `tol` or `learning_rate` properly: for example, `tol=0.0003` and `learning_rate=0.7` should work fine in this case.)

The multilayer inference finishes 3 independent runs in roughly 8 minutes with maximum 1.6 GB GPU memory usage. With the randomization seed of respectively 0, 1, and 2, the inference stops after 1978, 1679, and 2621 iterations, reaching a `pi` accuracy of 0.93, 0.92, and 0.82, and an `alpha` correlation of 0.52, 0.56, and 0.48.

## Additional Notes
1. The current network generation code we publish only supports layer overlap values of 0 or 1, since we were not able to come up with an elegant way to generate a network with an exact layer overlap rate between 0 and 1 (e.g., 0.5), while there exist very different solutions for approximating one (e.g., the generated network might have 0.4 or 0.6 layer overlap with the input parameter of 0.5). We encourage the users to implement this feature on their own if needed, based on their preference of the generation dynamics. For example, you can use the [`random_rewire`](https://graph-tool.skewed.de/static/doc/generation.html#graph_tool.generation.random_rewire) function from the [`graph-tool`](https://graph-tool.skewed.de/) library, as we did in our testing.
2. In our log generation code, we accelerate the process with multiprocessing. The cascades are generated in batches, and the cascades in each batch are generated in parallel. By default, the size of each batch is set to the number of CPU cores available on the machine.
3. A cascade generated through the SIR process can be empty (i.e., only the seed node is activated and no spreading is observed). Since we exclude such uninformative cascades for the inference, the users would need to generate more cascades than the number they expect to have in the inference process. For example, under the setting in our provided code, we generate `100*E` cascades and get roughly `20*E` nonempty cascades that are usable in the inference. Note that the proportion of nonempty cascades varies significantly with the setting of the parameters (especially the recovery rate parameter `gamma`), and also slightly with each run of the log generation process.
4. The generated spreading logs are saved in a pickle file as a Python dictionary, where the key is the index of the cascade, and the value is a list of activation logs, each in the format of `(t, u, j)`, representing a cascade spreading from node `u` to node `j` at time `t`. The information of the exact activator `u` is not used in the inference because it is usually absent in real spreading data.
5. The inference code (for both phases) can be run on either a GPU or a CPU. The code by default uses the GPU if it detects a usable GPU on the machine, or uses the CPU otherwise. The memory limit and the running time of the code depends heavily on the hardware setting.
6. One advantage of the two-phase inference process is that the objective function of the single layer phase is convex, although that of the multilayer phase is not. Therefore, to deal with this nonconvexity, you don't need to run the single layer inference multiple times (which is usually most time-consuming), but you can finish multiple runs of the multilayer inference within relatively manageable time (especially when you have increased the cascade size threshold `s_c` properly).
7. We currently do not provide the code for applying the inference to real spreading data, although it is not too difficult to create one by modifying our current code. The main reason is that we have spotted certain flaws in the current inference method that potentially lead to misleading interpretations of the inference results. We are actively working to improve the inference method accordingly.
8. Please contact [me](mailto:yan.xia@aalto.fi) if you have any other problem with using the code, or any suggestion of improvement.

## Citation
Please cite the following paper if you are using this code:
```
@article{xia2022limits, 
  title={Limits of Multilayer Diffusion Network Inference in Social Media Research}, 
  author={Xia, Yan and Chen, Ted Hsuan Yun and Kivel{\"a}, Mikko}, 
  journal={Proceedings of the International AAAI Conference on Web and Social Media}, 
  volume={16}, 
  number={1}, 
  pages={1145-1156},
  year={2022}, 
  month={May}
}
```
