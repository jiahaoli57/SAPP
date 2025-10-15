# SAPP: Sparse Attention Transformers for Pattern Recognition in Portfolio Management

<div align="center">
<img align="center" src=figures/SAPP.jpg width="40%"/> 

<div>&nbsp;</div>

[![Python 3.9](https://shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31016/)
[![Read the Docs](https://readthedocs.org/projects/optuna/badge/?version=latest)](https://finol.readthedocs.io/en/latest/)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macos-lightgrey)](Platform)
[![License](https://img.shields.io/github/license/jiahaoli57/FinOL)](LICENSE)

</div>

## Baseline Implementations

In our paper, we benchmarked SAPP against a comprehensive set of baseline methods, including classic, state-of-the-art, and Transformer-based models. As per the Journal's request, we provide access to the baseline code as follows:

### Reproduced Baselines

For the following two Transformer-based models, we have included our own implementation within this repository to ensure a fair and consistent comparison. You can find their code in the `finol/model_layer` directory.

<details open>
<summary><strong>List of the Transformer-based baselines we implemented</strong></summary>


* **Sparse Transformer**: Child et al. 2019, _arXiv_ [[code](https://github.com/jiahaoli57/SAPP/tree/main/finol/model_layer/SparseTransformer.py)] [[paper](https://arxiv.org/abs/1904.10509)]
* **DRSA**: Wang et al. 2025, _Pattern Recognition_ [[code](https://github.com/jiahaoli57/SAPP/tree/main/finol/model_layer/DRSA.py)] [[paper](https://www.sciencedirect.com/science/article/pii/S0031320324008094)]


</details>

### Referenced Baselines

For most of the classic and state-of-the-art online portfolio selection strategies, we refer to existing implementations in well-established open-source libraries.

A majority of the following algorithms from our study can be found in the excellent `olps` Matlab library:

> **GitHub - OLPS**: [https://github.com/OLPS/OLPS](https://github.com/OLPS/OLPS)

<details open>
<summary><strong>List of the specific baselines we compared against</strong></summary>


* **Benchmark Baselines**:
  * **UCRP**: Kelly 1956, _The Bell System Technical Journal_ [[code](https://github.com/OLPS/OLPS/blob/master/Strategy/ucrp_run.m)] [[paper](https://ieeexplore.ieee.org/abstract/document/6771227/)]
  * **BCRP**: Cover 1991, _Mathematical Finance_ [[code](https://github.com/OLPS/OLPS/blob/master/Strategy/bcrp_run.m)] [[paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9965.1991.tb00002.x)]
* **Classic Baselines**:
  * **UP**: Cover 1991, _Mathematical Finance_ [[code](https://github.com/OLPS/OLPS/blob/master/Strategy/up_run.m)] [[paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9965.1991.tb00002.x)]
  * **EG**: Helmbold et al. 1998, _Mathematical Finance_ [[code](https://github.com/OLPS/OLPS/blob/master/Strategy/eg_run.m)] [[paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-9965.00058)]
  * **ONS**: Agarwal et al. 2006,	_International Conference on Machine Learning_ [[code](https://github.com/OLPS/OLPS/blob/master/Strategy/ons_run.m)] [[paper](https://dl.acm.org/doi/abs/10.1145/1143844.1143846)]
* **State-of-the-art Baselines**:
  * **ANTI<sup>1</sup>**: Borodin et al. 2004, _Advances in Neural Information Processing Systems_ [[code](https://github.com/OLPS/OLPS/blob/master/Strategy/anticor_run.m)] [[paper](https://proceedings.neurips.cc/paper_files/paper/2003/hash/8c9f32e03aeb2e3000825c8c875c4edd-Abstract.html)]
  * **ANTI<sup>2</sup>**: Borodin et al. 2004, _Advances in Neural Information Processing Systems_ [[code](https://github.com/OLPS/OLPS/blob/master/Strategy/anticor_anticor_run.m)] [[paper](https://proceedings.neurips.cc/paper_files/paper/2003/hash/8c9f32e03aeb2e3000825c8c875c4edd-Abstract.html)] 
  * **PAMR**: Li et al. 2012,	_Machine Learning_ [[code](https://github.com/OLPS/OLPS/blob/master/Strategy/pamr_run.m)] [[paper](https://link.springer.com/article/10.1007/s10994-012-5281-z)]
  * **CWMR-Var**: Li et al. 2013,	_ACM Transactions on Knowledge Discovery from Data_ [[code](https://github.com/OLPS/OLPS/blob/master/Strategy/cwmr_var_run.m)] [[paper](https://link.springer.com/article/10.1007/s10994-012-5281-z)]
  * **CWMR-Stdev**: Li et al. 2013,	_ACM Transactions on Knowledge Discovery from Data_ [[code](https://github.com/OLPS/OLPS/blob/master/Strategy/cwmr_stdev_run.m)] [[paper](https://link.springer.com/article/10.1007/s10994-012-5281-z)]
  * **OLMAR-S**: Li et al. 2015,	_Artificial Intelligence_ [[code](https://github.com/OLPS/OLPS/blob/master/Strategy/olmar1_run.m)] [[paper](https://www.sciencedirect.com/science/article/pii/S0004370215000168)]
  * **OLMAR-E**: Li et al. 2015,	_Artificial Intelligence_ [[code](https://github.com/OLPS/OLPS/blob/master/Strategy/olmar2_run.m)] [[paper](https://www.sciencedirect.com/science/article/pii/S0004370215000168)]
  * **RMR**: Huang et al. 2016,	_IEEE Transactions on Knowledge and Data Engineering_ [[code](https://github.com/OLPS/OLPS/blob/master/Strategy/rmr_run.m)] [[paper](https://ieeexplore.ieee.org/abstract/document/7465840)]
</details>


### Official Implementations for Specific SOTA Baselines
For some of the more recent state-of-the-art baselines not included in the general `olps` toolkit, we link directly to the official code provided by the authors:

<details open>
<summary><strong>List of the specific baselines we compared against</strong></summary>


* **SPOLC**: Lai et al. 2020, _Journal of Machine Learning Research_ [[code](https://github.com/laizhr/SPOLC/blob/master/SPOLC_run.m)] [[paper](https://www.jmlr.org/papers/v21/19-959.html)]
* **RPRT**: Lai et al. 2020, _IEEE Transactions on Systems, Man, and Cybernetics: Systems_ [[code](https://github.com/laizhr/RPRT/blob/master/RPRT_run.m)] [[paper](https://ieeexplore.ieee.org/abstract/document/8411138)]

</details>


## Code Availability Note⏳
The code for this project (SAPP: Sparse Attention Transformers for Pattern Recognition in Portfolio Management) is currently being prepared for public release. As our team has just completed the submission of the revised manuscript, we require 1-2 days to finalize code checks and organize instructions.

## License

Released under the [MIT License](https://github.com/jiahaoli57/sapp/blob/main/LICENSE).

## Contact Us

For further discussions, please get in touch with the repo manager (Jiahao Li) via lijh@pbcsf.tsinghua.edu.cn.
