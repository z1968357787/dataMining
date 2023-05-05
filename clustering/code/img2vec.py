import numpy as np
from sklearn import preprocessing
from skimage import io, transform
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# 读取原始图片
path = r'E:\University\onedrive\OneDrive - mail.scut.edu.cn\助教\数据挖掘\第一次实验\参考图片'
img0 = io.imread(path + '\\scut.jpg')
print('img0彩色图像的形状为：', img0.shape)
img_gray = rgb2gray(img0)
rescale_img = transform.rescale(img_gray, 0.5)

print('灰度图像的形状为：', rescale_img.shape)
binarizer = preprocessing.Binarizer(threshold=.3).fit(rescale_img)
img_gray_binary = binarizer.transform(rescale_img)
io.imshow(img_gray_binary)

# 获得灰度值为0的元素索引
scatter_idx = np.array(np.where(img_gray_binary == 0))
print(scatter_idx.shape)
# 绘制灰度值为0的元素散点图
plt.figure(figsize=(10, 10), frameon=False)
plt.scatter(scatter_idx[1], -scatter_idx[0], marker='.')  # 显示灰度图像
plt.show()

x = scatter_idx[1]
y = -scatter_idx[0]
output_points = np.concatenate(([x], [y]), axis=0).T
print(output_points.shape)
print("first_point:", output_points)
np.save("scutVec.npy",output_points)