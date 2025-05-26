import os
import cv2
import numpy as np
from ultralytics import YOLO
import time

# 1. 路径配置

# RGB图路径
image_dir = 'F:\\pythonProject1\\wolfberry_dataset\\images\\test'
# 深度图路径
depth_dir = r'D:\BaiduNetdiskDownload\20250109_gouqi_mock_lab\save_20250119_172516_953198\depth_images'
# 输出文件夹根目录
output_base_dir = 'F:\\pythonProject1\\wolfberry_results'

# 2. 创建时间戳输出目录
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(output_base_dir, timestamp)
os.makedirs(output_dir, exist_ok=True)

# 3. 加载 YOLOv8 模型（检测枸杞）
model_path = 'runs/detect/train48/weights/best.pt'
model = YOLO(model_path)

# 4. 读取RAW深度图（.raw格式 → 2D深度矩阵）
def read_raw_depth_img(depth_fname, height=1080, width=1920, scale=1000):
    depth_data = np.fromfile(depth_fname, dtype=np.uint16).astype(float) / scale
    return depth_data.reshape((height, width))

# 5. 判断像素是否为红色（HSV范围）
def is_red_pixel_hsv(hsv_pixel):
    h, s, v = hsv_pixel
    return ((0 <= h <= 10 or 160 <= h <= 180) and s >= 100 and v >= 50)

# 6. 区域生长算法（颜色+深度约束）
def region_grow_depth_color_hsv(hsv_img, depth_map, center, depth_thresh=0.03):
    h, w = depth_map.shape
    visited = np.zeros((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.uint8)
    queue = [center]
    base_depth = depth_map[center[1], center[0]]  # 以中心点深度为基准

    while queue:
        x, y = queue.pop(0)
        if visited[y, x]:
            continue
        visited[y, x] = True
        current_depth = depth_map[y, x]
        current_color = hsv_img[y, x]
        # 同时满足颜色和深度差约束
        if abs(current_depth - base_depth) < depth_thresh and is_red_pixel_hsv(current_color):
            mask[y, x] = 255
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    queue.append((nx, ny))
    return mask

# 7. 判断候选框是否重复（靠近已有框且深度相近）
def is_duplicate(new_box, existing_boxes, new_depth, depth_map):
    nx, ny = int((new_box[0] + new_box[2]) / 2), int((new_box[1] + new_box[3]) / 2)
    for box in existing_boxes:
        ex, ey = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        if np.linalg.norm([nx - ex, ny - ey]) < 20:
            existing_depth = depth_map[ey, ex]
            if abs(existing_depth - new_depth) < 0.1:
                return True
    return False

# 8. 主处理流程
for filename in os.listdir(image_dir):
    if not filename.endswith('.jpg'):
        continue

    # 读取图像和对应深度图
    image_path = os.path.join(image_dir, filename)
    img = cv2.imread(image_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    name_prefix = filename.replace('_color.jpg', '')
    depth_path = os.path.join(depth_dir, name_prefix + '_depth.raw')
    depth_map = read_raw_depth_img(depth_path)

    # YOLO检测
    results = model(img)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    boxes_confs = sorted(zip(boxes, confs), key=lambda x: -x[1])  # 置信度排序

    # 初始化输出数据结构
    kept_boxes = []
    output_img = img.copy()
    mask_output = np.zeros(img.shape[:2], dtype=np.uint8)
    contour_img = img.copy()
    used_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    box_id = 0
    for box, conf in boxes_confs:
        x1, y1, x2, y2 = map(int, box)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if cx >= depth_map.shape[1] or cy >= depth_map.shape[0]:
            continue
        center_depth = depth_map[cy, cx]
        if is_duplicate([x1, y1, x2, y2], kept_boxes, center_depth, depth_map):
            continue

        kept_boxes.append([x1, y1, x2, y2])

        # 绘制检测框
        color = (0, 255, 0) if conf >= 0.5 else (0, 255, 255)
        cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
        label = f"{box_id} ({conf:.2f})"
        cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 保存深度txt
        sx1, sy1 = max(0, x1 - 20), max(0, y1 - 20)
        sx2, sy2 = min(depth_map.shape[1], x2 + 20), min(depth_map.shape[0], y2 + 20)
        depth_crop = depth_map[sy1:sy2, sx1:sx2]
        txt_dir = os.path.join(output_dir, 'depth')
        os.makedirs(txt_dir, exist_ok=True)
        np.savetxt(os.path.join(txt_dir, f"{name_prefix}_box{box_id}.txt"), depth_crop, fmt="%.3f")

        # 区域生长 & 获取可见区域
        mask = region_grow_depth_color_hsv(hsv_img, depth_map, (cx, cy))
        visible_mask = cv2.bitwise_and(mask, cv2.bitwise_not(used_mask))
        used_mask = cv2.bitwise_or(used_mask, mask)
        mask_output = cv2.bitwise_or(mask_output, visible_mask)

        # 获取轮廓
        contours, _ = cv2.findContours(visible_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 10:
                continue
            # 画绿色实线轮廓
            cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 1)


            # if len(cnt) >= 5:
            #     ellipse = cv2.fitEllipse(cnt)
            #     cv2.ellipse(contour_img, ellipse, (0, 255, 0), 1)  # 绿色椭圆线条
            # else:
            #     cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 1)  # 少于5点无法拟合，退回轮廓线绘制

            # 在轮廓内提取所有点及其深度
            ys, xs = np.where(visible_mask > 0)
            depth_vals = []
            for x, y in zip(xs, ys):
                if cv2.pointPolygonTest(cnt, (float(x), float(y)), False) >= 0:
                    d = depth_map[y, x]
                    depth_vals.append((x, y, d))

            if len(depth_vals) == 0:
                continue

            # 以深度升序排序，找到最浅的前若干点
            depth_vals_sorted = sorted(depth_vals, key=lambda item: item[2])
            shallowest_points = depth_vals_sorted[:200]  # 可调整：取最浅的前200个点

            # 计算这些点的中心位置（用于选取30x30区域）
            xs_shallow = [pt[0] for pt in shallowest_points]
            ys_shallow = [pt[1] for pt in shallowest_points]
            mean_x = int(np.mean(xs_shallow))
            mean_y = int(np.mean(ys_shallow))

            # 在原图上绘制一个 10x10 浅绿色区域
            half_box = 5
            tl_x, tl_y = max(mean_x - half_box, 0), max(mean_y - half_box, 0)
            br_x, br_y = min(mean_x + half_box, contour_img.shape[1]), min(mean_y + half_box, contour_img.shape[0])

            # 定义透明度 alpha（0~1）
            alpha = 0.4  # 透明度：0=完全透明，1=完全不透明

            # 创建 overlay 层，并画上浅绿色矩形
            overlay = contour_img.copy()
            cv2.rectangle(overlay, (tl_x, tl_y), (br_x, br_y), (0, 255, 180), thickness=-1)  # BGR颜色填充

            # 将 overlay 混合到 contour_img 上（只对指定区域）
            contour_img = cv2.addWeighted(overlay, alpha, contour_img, 1 - alpha, 0)

        box_id += 1

    # 保存输出图像（检测图、掩码图、轮廓图）
    cv2.imwrite(os.path.join(output_dir, f"{name_prefix}_boxes.jpg"), output_img)
    cv2.imwrite(os.path.join(output_dir, f"{name_prefix}_mask.png"), mask_output)
    cv2.imwrite(os.path.join(output_dir, f"{name_prefix}_contour.jpg"), contour_img)
