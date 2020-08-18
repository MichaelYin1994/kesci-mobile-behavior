## 2020新网银行金融科技挑战赛 AI算法赛道

---
### 队伍简介
队名：我们的队伍向太阳，队长：鱼丸粗面(zhuoyin94@163.com)。初赛A榜榜上排名第14，线上分数0.80590476；初赛B榜榜上排名第35（更新，去小号后排名第16），线上分数0.78702646。（20200711开始）

---
### 系统环境与主要依赖packages
- 系统环境: Ubuntu 20.04 LTS
- GPU: Quadro GV100(32G) × 2
- CPU: Intel® Xeon(R) Gold 6138 CPU @ 2.00GHz × 80
- RAM: 256G
- python: 3.6.9
- tensorflow-gpu: 2.1.0
- gensim: 3.8.1
- sklearn: 0.23.1
- pandas: 1.0.4
  
---
### 基本思路说明
本项目采用了深度学习端到端的思路对用户行为时序进行建模。整个项目分为如下几个模块：

- **预处理与数据增强**: 首先通过FFT采样，将每一条用户传感器行为时序采样至等长；随后采用文献[1]的思路对训练数据进行增强，对验证数据保持不变，选择最优的参数。在该过程中，我们也尝试使用Kalman Filter[3]对数据噪声进行滤除，线上效果不佳。

- **神经网络结构1**: 对应文件nn_v1.py与nn_v2.py，参考了项目[6]的相关思路，但是结构彻底改变。参考Multi-head Attention策略[1]、TextCNN结构[7]与ResNet结构[8]，构建了多Kernel的2D-CNN网络，2D卷积的应用跨特征的抽取了时序的特性；ResNet结构辅助防止了过拟合；多Kernel的结构抽取了不同的卷积特征。经过多层神经网络和average pooling最终进行分类。

- **神经网络结构2**: 对应文件nn_v3.py，实现上参考了项目[5]的相关思路与TextCNN思路[7]以及ResNet结构[8]。构建针对CONV-1D的ResNet结构，随后级联构建网络。CONV-1D的使用从时序层面抽取了特征；多Kernel也从不同的角度抽取了特征；而ResNet结构辅助防止了过拟合，分类层结构与神经网络结构1类似，此处不再赘述。

- **堆叠集成**: 针对3种结构的神经网络，最终采用简单平均的方法，融合不同神经网络的分类结果。

---
### 文件结构
文件目录如下：
- **.//plots//**: 用于存储神经网络训练过程中每一fold的loss curve与accuracy curve，并采用S-G滤波进行平滑。
- **.//submissions//**: 用于存储当前神经网络的运行结果。
- **.//submission_oof//**: 用于存储神经网络训练过程中每一fold的out of fold预测结果。
- **.//data//**: 用于存储原始官方数据。
- **.//data_tmp//**: 用于存储预处理后的数据与一些临时数据。

---
### 代码运行说明
代码运行顺序：
- **preprocessing.py**: 从.//data//读取官方数据进行预处理，生成暂存文件存放于.//data_tmp//。
- **nn_v1.py**: 从.//data_tmp//读取数据进行训练。训练完成后生成的可提交结果存储于.//submissions//路径下，out of fold的结果存储于.//submission_oof//路径下。
- **nn_v2.py**: 同上。
- **nn_v3.py**: 同上。
- **stacking_oof.py**: 读取nn_v1.py、nn_v2.py与nn_v3.py的oof结果，进行简单平均集成，并存储集成结果到.//submissions//文件夹，作为最终提交结果。

运行注意事项：

1. 若神经网络运行出现Out Of Memory错误，请降低211行左右的BATCH_SIZE关键字。
2. 在三组nn训练完之后，需要手动更新stacking_oof.py中的三组oof的路径，读取其结果进行平均。
3. 神经网络的训练可以3张卡在Pycharm IDE用Docker同时进行，需要修改第49行左右GPU设备的ID号。
4. 附带项目github地址: https://github.com/MichaelYin1994/kesci-mobile-behavior

---
### References
[1] Fawaz H I, Forestier G, Weber J, et al. Data augmentation using synthetic data for time series classification with deep residual networks[J]. arXiv preprint arXiv:1808.02455, 2018.

[2] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need[C]//Advances in neural information processing systems. 2017: 5998-6008.

[3] Greg Welch, Gary Bishop. An Introduction to the Kalman Filter[M]. University of North Carolina at Chapel Hill, 1995.

[4] Rakthanmanon T, Campana B, Mueen A, et al. Searching and mining trillions of time series subsequences under dynamic time warping[C]//Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining. 2012: 262-270.

[5] CONV-1D Baseline: https://github.com/blueloveTH/xwbank2020_baseline_keras

[6] CONV-2D Baseline: https://github.com/ycd2016/xw2020_cnn_baseline

[7] Kim Y. Convolutional neural networks for sentence classification[J]. arXiv preprint arXiv:1408.5882, 2014.

[8] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

[9] （腾讯张戎，时序分类网络Review）https://zhuanlan.zhihu.com/p/83130649

[10] （时序数据增强策略代码）https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
