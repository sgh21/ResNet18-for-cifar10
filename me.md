#### 演示视频说明
+ 通过python ModelTest.py进行模型测试
+ 识别出错时自动弹出图片，可放大查看
+ x掉图片可以，按任意建继续执行
+ 关掉图片输出模式，查看全部图片的测试结果
+ 看到训练集50000张图片，准确率99%
+ 测试测试集的准确率
+ 看到测试集10000张图片，准确率93%
#### Environment
+ python 3.10
+ pytorch 2.2.0
+ torchvision 0.11.1
+ cuda 12.1
+ matplotlib 3.1.3
+ numpy 1.16.4
#### Get started
```python
conda install  python=3.10 pytorch=2.2.0 torchvision=0.11.1 cudatoolkit=12.1 matplotlib=3.1.3 numpy=1.16.4
```
#### 运行
```python
python ModelTest.py
```