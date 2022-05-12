# BISHE-PRODEN
渐进识别的偏标记学习算法包 *By Jiewang*

包括PRODEN算法、三种变体算法以及监督学习模式的情况

## 运行说明
运行时只需要在命令行输入类似代码即可运行：
```
python main.py -ds mnist -model linear -partial_type binomial -partial_rate 0.1
 ```
ds指的是数据集，model指的是基础模型，partial_type指的是翻转模式，本实验所用的是二项分布翻转模式，partial_rate指的是标记被翻转概率。

