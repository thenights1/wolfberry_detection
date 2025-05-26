import os
import torch
from ultralytics import YOLO

# 选择训练设备（优先使用GPU）
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 加载YOLOv8的轻量模型（Nano版）
model = YOLO("./yolov8n.pt").to(device)

if __name__ == '__main__':
    # 开始训练
    results = model.train(
        data='data.yaml',   # 数据集配置文件
        epochs=50,          # 训练轮数
        imgsz=512,          # 图像输入大小
        batch=16,           # 每轮训练的批次数
        device=device,      # 使用的设备（GPU或CPU）
        augment=True        # 是否开启数据增强
    )

    # 在验证集上评估模型效果
    print("\\nEvaluating on Validation Set...")
    val_metrics = model.val(split='val', augment=True)  # 启用TTA增强
    print(val_metrics)

    # 在测试集上评估效果
    print("\\nEvaluating on Test Set...")
    test_metrics = model.val(split='test', augment=True)
    print(test_metrics)

    # 输出平均精度指标
    print("\\nValidation mAP50:", val_metrics.box.map50)
    print("Validation mAP50-95:", val_metrics.box.map)
    print("Test mAP50:", test_metrics.box.map50)
    print("Test mAP50-95:", test_metrics.box.map)

    # 对测试集中的图像进行推理并保存
    test_images_path = "F:/pythonProject1/wolfberry_dataset/images/test"
    save_results_dir = "F:/pythonProject1/wolfberry_results"
    os.makedirs(save_results_dir, exist_ok=True)

    print("\\nTesting on single images...")
    for img_name in os.listdir(test_images_path):
        img_path = os.path.join(test_images_path, img_name)
        results = model(img_path, save=True, conf=0.5)
        print(f"Processed {img_name}, results saved.")

    print("\\nTesting completed. Check results folder.")
