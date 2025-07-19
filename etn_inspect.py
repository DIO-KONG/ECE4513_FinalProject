import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from models import MiniExpressionTransformNet

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MiniExpressionTransformNet()
model.load_state_dict(torch.load('models/model_etjaffe.pth'))
model.to(device)
model.eval()

# 加载图片并预处理
image_path = 'data/jaffe_crop/KA.DI2.43.tiff'  # 替换为你的图片路径
image = Image.open(image_path).convert('L')  # 转为灰度图
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
input_image = transform(image).unsqueeze(0).to(device)  # 添加batch维度

# 模型推理
with torch.no_grad():
    output_image = model(input_image)

# 将输出转换为可视化格式
output_image = output_image.squeeze(0).cpu().numpy()  # 移除batch维度

# 可视化原图和输出
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Transformed Image")
plt.imshow(output_image[0], cmap='gray')  # 显示第一个通道
plt.axis('off')

plt.show()