# CNN Webshell 检测模型

通过卷积神经网络来检测检测 PHP 恶意脚本的一次尝试，欢迎交流~

## 安装说明

安装依赖：

```
pip install -r requirements
```

初始化数据集：

```
git submodule init
git submodule update
```

初始化数据库（用于 Demo，可选）：

```
mysql -u<username> -p<password> < schema.sql
```

## 使用说明

训练新模型：

```
./training.py
```

运行 Demo（默认绑定在 `0.0.0.0:5000`）：

```
./demo.py
```

测试已有模型（位于 `persistence` 下）：

```
./test_model_metric_exist.py
```

训练新模型，并对其进行测试：

```
./test_model_metric_new.py
```

训练 RNN 模型，并对其进行测试：

```
./test_model_metric_rnn.py
```

## 参考资料

1. [基于 CNN 的 Webshell 检测平台的设计与实现](https://www.grassfish.net/2017/11/18/cnn-webshell-detect/)
2. [基于机器学习的 Webshell 发现技术探索](https://segmentfault.com/a/1190000011112448)
