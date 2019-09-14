# A comparison of some conformal quantile regression methods

We compare two recently proposed methods that combine ideas from conformal inference and
quantile regression to produce locally adaptive and marginally valid prediction intervals under
sample exchangeability.

Accompanying paper:

 - Matteo Sesia and Emmanuel J. Candes, "A comparison of some conformal quantile regression methods", 2019. [arXiv:1909.05433](https://arxiv.org/abs/1909.05433)

The methods we are comparing are described in:

 - Yaniv Romano, Evan Patterson, and Emmanuel J. Candes, "Conformalized quantile regression", 2019.
 
 - Danijel Kivaranovic, Kory D. Johnson, and Hannes Leeb, "Adaptive, Distribution-Free Prediction Intervals for Deep Neural Networks", 2019.

The code in this repository is a fork of [https://github.com/yromano/cqr](https://github.com/yromano/cqr).

### Prerequisites

* python
* numpy
* scipy
* scikit-learn
* scikit-garden
* pytorch

### Installing

The development version is available here on GitHub:
```bash
git clone https://github.com/msesia/cqr-comparison.git
```

## Reproducible Research

The code available under 'experiments/' in the repository replicates the experimental results in our paper.

### Publicly Available Datasets

* [Bike](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset): Bike sharing dataset data set.

* [Bio](https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure): Physicochemical properties of protein tertiary structure data set.

* [Blog](https://archive.ics.uci.edu/ml/datasets/BlogFeedback): BlogFeedback data set.

* [Community](http://archive.ics.uci.edu/ml/datasets/communities+and+crime): Communities and crime data set.

* [Concrete](http://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength): Concrete compressive strength data set.

* [Facebook Variant 1 and Variant 2](https://archive.ics.uci.edu/ml/datasets/Facebook+Comment+Volume+Dataset): Facebook comment volume data set.

* [Homes](https://www.kaggle.com/harlfoxem/housesalesprediction): House sale prices for King County.

* [STAR](https://www.rdocumentation.org/packages/AER/versions/1.2-6/topics/STAR): C.M. Achilles, Helen Pate Bain, Fred Bellott, Jayne Boyd-Zaharias, Jeremy Finn, John Folger, John Johnston, and Elizabeth Word. Tennessee’s Student Teacher Achievement Ratio (STAR) project, 2008.


### Data subject to copyright/usage rules

The Medical Expenditure Panel Survey (MPES) data can be downloaded using the code in the folder /get_meps_data/ under this repository. It is based on [this explanation](https://github.com/yromano/cqr/blob/master/get_meps_data/README.md) (code provided by [IBM's AIF360](https://github.com/IBM/AIF360)).

* [MEPS_19](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181): Medical expenditure panel survey,  panel 19.

* [MEPS_20](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-181): Medical expenditure panel survey,  panel 20.

* [MEPS_21](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192): Medical expenditure panel survey,  panel 21.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
