# 数据说明
original: 包含5000余张彩色图片,大小160*160,无噪声</br>
add_noise.py: 给original中的图片添加噪声后,保存于noisy文件夹的同名图片。</br>
              实际训练和测试的数据不需要这样生成,其噪声是数据加载时随机产生的(见dataset.py)</br>
data_split.py:将original中图片的文件名按比例分成两部分,写入 trainset.txt 和 testset.txt。</br>
