# CNN Webshell 检测

## 安装

第三方库：

```
pip install -r requirements
```

初始化数据库：

```
mysql -u<username> -p<password> < init.sql
```

## 使用

模型训练：

```
./training.py
```

测试已有模型：

```
./test_model_metric_exist.py
```

训练新模型并测试：

```
./test_model_metric_new.py
```

测试 RNN 模型：

```
./test_model_metric_rnn.py
```

运行检测 Demo 页面：

```
./server.py
```
