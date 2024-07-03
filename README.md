## MSPredictor: A Multi-Scale Dynamic Graph Neural Network for Multivariate Time Series Prediction

### Abstract

> **In the field of multivariate time series forecasting, capturing the dynamic relationships and complex cyclical pat- terns between sequences is key to improving prediction accuracy. To address this challenge, our paper introduces MSPredictor, a multi-scale dynamic graph neural network model, which uses Fast Fourier Transform (FFT) for multi-scale decoupling in the frequency domain and employs Kolmogorov-Arnold Networks (KANs) for multi-scale fusion, effectively extracting significant cyclical patterns. By decomposing the original series across different scales, MSPredictor accurately models complex cyclical patterns. To enhance the model’s transparency and interpretabil- ity, we introduced the ClarityLens explanatory strategy, which employs visualization techniques to make the prediction process more transparent. Specifically, it displays the adjacency matrices learned at different scales, intuitively showing the dynamic correlations between series. We also visualized the proportion of different periods in the prediction results and the specific forecasting performance at each time scale. Extensive testing on multiple real-world datasets has demonstrated that the MSPre- dictor significantly outperforms existing benchmarks, validating its practicality and high transparency.**

Performance comparison of six selected methods in four types: GNNs, LLMs, Transformers, and Mixers, using the mean squared error as the metric.

![p1](imgs\p1.png)

Potential interactions among variables in MTS prediction are critical. Most studies have used a pre-set static correlation (s0). However, in reality, the graph structure changes over time (s1 and s2), and these changes differ based on the scale of observation (s3). Therefore, it is crucial to consider the dynamic nature and scale effects of these inter-variable interactions when predicting MTS.

![p2](imgs\p2.png)

Architecture of the MSPredictor. (a) Multi-Scale Decoupling Module (MDM): This module uses FFT to decompose the original sequence into different scales, capturing various periodic information in the frequency domain of the input sequence. (b) Evolving Graph Structure Learner (EGSL): This learner is responsible for learning and updating the multi-scale temporal graph structure to adapt to the dynamic changes in time series data. (c) Multi-Scale Spatiotemporal Module (MSTM): This module contains *k* GNNs and TCNs, designed to capture the dynamic and complex relationships between variables at specific scales. (d) Multi-Scale Fusion Module (MFM): This module integrates features and temporal pattern information from the MSTM. It effectively combines information from different scales through an *L*-layer KAN, significantly enhancing the accuracy and stability of predictions.

![p3](imgs\p3.png)

The architecture of multi-scale fusion module, which primarily en- compasses three key operations: (a) Concatenation is employed to amalgamate representations from various scales into a singular, unified vector, ensuring a comprehensive aggregation of information. (b) Pooling is applied to reduce the dimensionality of the combined feature vector, highlighting the most crucial features. (c) Fully connected layer composed of *L*-layer KAN.

![p4](imgs\p4.png)