import argparse
from distutils.util import strtobool
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math
from src.model import PConvUNet


def main(args):
    # Define the used device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Define the model
    print("Loading the Model...")
    model = PConvUNet(finetune=False, layer_size=7)
    model.load_state_dict(torch.load(args.model, map_location=device)['model'])
    model.to(device)
    model.eval()

    # Loading Input and Mask
    print("Loading the inputs...")
    org = Image.open(args.img)
    org = TF.to_tensor(org.convert('RGB'))
    mask = Image.open(args.mask)
    mask = TF.to_tensor(mask.convert('RGB'))
    inp = org * mask

    # Model prediction
    print("Model Prediction...")
    with torch.no_grad():
        inp_ = inp.unsqueeze(0).to(device)
        mask_ = mask.unsqueeze(0).to(device)
        if args.resize:
            org_size = inp_.shape[-2:]
            inp_ = F.interpolate(inp_, size=256)
            mask_ = F.interpolate(mask_, size=256)
        raw_out, _ = model(inp_, mask_)
    if args.resize:
        raw_out = F.interpolate(raw_out, size=org_size)

    # Post process
    raw_out = raw_out.to(torch.device('cpu')).squeeze()
    raw_out = raw_out.clamp(0.0, 1.0)
    out = mask * inp + (1 - mask) * raw_out

    # Saving an output image
    print("Saving the output...")
    out = TF.to_pil_image(out)
    img_name = args.img.split('/')[-1]
    out.save(os.path.join("examples", "out_{}".format(img_name)))


if __name__ == "__main__":
    # 加载图像
    img1 = Image.open('examples/image0.jpg').convert('RGB')
    img2 = Image.open('examples/out_image0.jpg').convert('RGB')

    # 转为 numpy 数组
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # 计算绝对差值
    diff = np.abs(arr1 - arr2)

    # # 参数设置
    # width, height = 256, 256
    # circle_radius = 5
    # num_circles = 20
    # background_value = 255  # 白色背景
    # hole_value = 0  # 黑色孔洞
    # # 创建全白图像
    # mask = np.full((height, width), background_value, dtype=np.uint8)
    # # 计算网格布局（近似均匀分布）
    # rows = int(math.sqrt(num_circles))
    # cols = (num_circles + rows - 1) // rows
    # dx = width // cols
    # dy = height // rows
    # # 在每个格子中画一个圆
    # for i in range(rows):
    #     for j in range(cols):
    #         if i * cols + j >= num_circles:
    #             break
    #         cx = j * dx + dx // 2
    #         cy = i * dy + dy // 2
    #         # 绘制圆形（使用 NumPy 的广播机制）
    #         for y in range(max(0, cy - circle_radius), min(height, cy + circle_radius)):
    #             for x in range(max(0, cx - circle_radius), min(width, cx + circle_radius)):
    #                 if (x - cx) ** 2 + (y - cy) ** 2 <= circle_radius ** 2:
    #                     mask[y, x] = hole_value
    # # 转换为图像并保存
    # mask_image = Image.fromarray(mask)
    # mask_image.save('examples/MASK.png')
    # print("Mask 已生成并保存为 circular_hole_mask.png")


    # img = Image.open('examples/img0.jpg').convert('RGB')
    # mask = Image.open('examples/MASK.png').convert('L')  # 转为灰度图
    #
    # img_array = np.array(img)
    # mask_array = np.array(mask)
    # mask_binary = (mask_array == 255).astype(np.uint8)
    # mask_array_expanded = mask_binary[:, :, np.newaxis]
    #
    # img_array1 = img_array * mask_array_expanded
    # # 转回图像并保存
    # result = Image.fromarray(img_array1)
    # result.save('examples/image0.jpg', 'JPEG')


    parser = argparse.ArgumentParser(description="Specify the inputs")
    parser.add_argument('--img', type=str, default="examples/image0.jpg")
    parser.add_argument('--mask', type=str, default="examples/MASK.png")
    #parser.add_argument('--model', type=str, default="examples/iter_1000000.pth")
    parser.add_argument('--model', type=str, default="examples/pretrained_pconv.pth")
    parser.add_argument('--resize', type=strtobool, default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    
    main(args)
