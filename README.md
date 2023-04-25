# Mixtures of Gaussian process experts based on kernel stick-breaking processes

by [Yuji Saikai](https://yujisaikai.com) and [Khue-Dung Dang](https://scholar.google.com.au/citations?user=lg56IkAAAAAJ&hl=en)

A preprint is (to be) on arXiv. The code was tested under Python `3.11.1` and NumPy `1.24.1`. To run a batch of experiments, execute `experiments.py`. You may want to change some parameters (e.g. number of training samples, MCMC iterations, etc.) at the top of the file. Results are save as `.pickle` files and store in `experiments' directory. To process the results, execute `analysis.py`.

**Abstract**
Mixtures of Gaussian process experts is a class of models that can simultaneously address two of the key limitations inherent in standard Gaussian processes: scalability and predictive performance. While the original model (Rasmussen and Ghahramani, 2001) is intuitive and capable of capturing non-stationarity, multi-modality and heteroskedasticity, its simplicity may limit the predictive performance when applied to complex data-generating processes. Capitalising on the recent advancement in the dependent Dirichlet processes literature, we propose a new mixture model of Gaussian process experts based on kernel stick-breaking processes. Our model maintains the intuitive appeal yet improve the performance of the original model. To make it practical, we design a sampler for posterior computation using the slice sampling. The model behaviour and improved predictive performance are demonstrated in extended experiments.

&nbsp;
![](GL2008_surface.png)

&nbsp;
![](GL2008_predictive_distributions.png)
