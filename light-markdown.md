# YOLOv5轻量化代码



## 1. 背景

原始的`detect.py`文件功能强大，支持多个检测模式（如批量图片、视频、摄像头流等），并且集成了多种后处理操作（如显示结果、保存检测结果等）。但过于屎山，这些功能对于许多实际应用来说可能是过度的，尤其是对资源要求严格的场合。

为了使YOLOv5更适用于这些场景，对原始代码进行简化和优化，保留了最基本的目标检测功能，并去除了一些不必要的部分。

## 2. 轻量化的目标

目标是尽量精简代码，保留以下核心功能：
- **加载YOLOv5模型**：能够加载训练好的模型并准备好进行推理。
- **图像检测**：支持从本地图片或摄像头流中进行目标检测。
- **推理和显示**：能够展示检测结果。

同时，需要去除：
- **多余的依赖**：去掉一些复杂的图像处理、结果保存等步骤。
- **不必要的选项**：例如，批量处理、模型优化、日志输出等非核心功能。

## 3. 轻量化步骤

### 3.1 精简YOLOv5模型加载部分

在原始的`detect.py`中，模型加载部分包括了很多选项和处理，如对输入的图片尺寸、推理时间的日志等。现只保留了最基本的部分来加载自定义YOLOv5模型：

```python
self.model = torch.hub.load('cra/yolov5', 'custom', path=model_path, source='local')  # 本地模型
```

省略了对模型超参数、配置的复杂处理，直接使用了最简单的本地加载方法。

### 3.2 删除无关的参数和功能

原始代码包含许多可以自定义的选项，如对检测结果的不同格式输出、推理过程的详细信息等。将这些去掉，只保留核心功能：

- **删除日志输出**：删除了模型推理时生成的日志和结果输出，只保留了错误检查和简单的控制台输出。
- **去除文件保存功能**：原始代码允许将检测结果保存为文件，现仅保留了在窗口中显示检测结果的功能。

```python
img_with_boxes = results.render()[0]
cv2.imshow('YOLOv5 Detection - Image', img_with_boxes)
```

### 3.3 只保留必要的推理部分

去除了原始代码中的一些不必要的推理步骤，如不同类型的后处理（如非最大抑制NMS），保留了最基础的模型推理功能。这保证了检测的核心流程得以简化，但同时不影响最终的检测效果。

### 3.4 删除批量处理支持

在YOLOv5的原始代码中，支持批量图片处理，但对于许多轻量级应用来说，我们只需要支持单张图片或视频流中的检测。因此，删去了批量处理相关的部分，并只保留了以下两种简单的模式：

- 从摄像头进行实时检测。
- 从本地图片进行检测。

```python
if choice == '1':
    detector.detect_from_camera(camera_index=0)  # 0表示本地默认摄像头
elif choice == '2':
    detector.detect_from_image(image_path)
```

### 3.5 简化视频流处理

原始代码还支持从视频文件读取并进行推理，现去除了视频文件输入部分，仅保留了从摄像头读取视频流并进行目标检测的功能。这使得代码更加专注于实时检测，而不需要处理文件输入输出。

```python
cap = cv2.VideoCapture(camera_index)
```

### 3.6 保留核心依赖

为了实现最基本的检测功能，保留了`torch`和`opencv-python`（`cv2`）这两个核心依赖。`torch`用于加载YOLOv5模型并进行推理，`cv2`用于图像加载和结果展示。

### 3.7 简洁用户输入
为了更加方便地选择检测模式，通过简单的`input`函数来进行选择。通过用户输入来选择摄像头模式或图片检测模式，程序便会进入相应的检测流程：

```python
choice = input("请输入选项 (1或2): ")
```

## 4. 代码

最后的轻量化YOLOv5检测代码：

```python
import cv2
import torch
import os

class YOLOv5Detector:
    def __init__(self, model_path, device='cpu'):
        """
        初始化YOLOv5模型
        :param model_path: 本地模型路径，例如 'yolov5s.pt'
        :param device: 运行设备 'cpu' 或 'cuda'
        """
        self.device = torch.device(device)
        self.model = torch.hub.load('cra/yolov5', 'custom', path=model_path, source='local')  # 本地模型
        self.model.to(self.device)
        self.model.eval()
        print("YOLOv5 模型加载成功！")

    def detect_from_camera(self, camera_index=0):
        """
        从本地摄像头进行检测
        :param camera_index: 本地摄像头的索引，默认0
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("无法打开本地摄像头，请检查设备！")
            return

        print("开始从本地摄像头读取...")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取视频帧")
                break

            # 推理
            results = self.model(frame)

            # 显示带有检测结果的画面
            img_with_boxes = results.render()[0]
            cv2.imshow('YOLOv5 Detection - Local Camera', img_with_boxes)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("摄像头检测结束！")

    def detect_from_image(self, image_path):
        """
        从本地图片进行检测
        :param image_path: 图片的绝对路径
        """
        if not os.path.isfile(image_path):
            print(f"文件路径 {image_path} 不存在，请检查！")
            return

        print(f"开始检测图片：{image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print("无法读取图片，请检查路径和文件格式！")
            return

        # 推理
        results = self.model(img)

        # 显示检测结果
        img_with_boxes = results.render()[0]
        cv2.imshow('YOLOv5 Detection - Image', img_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("图片检测结束！")

# 主程序入口
if __name__ == "__main__":
    model_path = 'yolov5s.pt'  # 模型路径，确保该文件存在本地

    detector = YOLOv5Detector(model_path=model_path, device='cpu')

    # 选择检测模式
    print("请选择检测模式：")
    print("1 - 使用本地摄像头")
    print("2 - 使用本地图片")

    choice = input("请输入选项 (1或2): ")

    if choice == '1':
        detector.detect_from_camera(camera_index=0)  # 0表示本地默认摄像头
    elif choice == '2':
        image_path = r"C:\Users\20601\Desktop\picture2\your_image.jpg"  # 实际的本地图片路径
        detector.detect_from_image(image_path)
    else:
        print("无效的选项！程序退出。")
```

## 5. 总结

主要保留了模型加载、图像或摄像头推理和检测结果展示的功能。。