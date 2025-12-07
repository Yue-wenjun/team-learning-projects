# pytorch-UNet-Denoise

### 介绍
Pytorch实现UNet, 去除彩色图片中的高斯噪声

### 运行环境
torch
torchvision
numpy
tqdm

### 使用说明
data: 存放图片数据</br>
weights: 保存模型</br>
train_image: 保存每轮训练结果, 噪声图、原图、去噪结果, 拼接形成的图片</br>
dataset.py: 数据集加载方式</br>
model.py: 模型结构</br>
train.py: 训练代码</br>
test.py: 测试代码,从测试集中选取图片,添加噪声,展示去噪效果</br>
color_map.jpg: 一张彩色图片;可能因为和数据集的图片风格差异巨大,模型对这张图片的效果不好</br>

### 参考资料

模型搭建参考: https://blog.csdn.net/keshi12354/article/details/147287355
