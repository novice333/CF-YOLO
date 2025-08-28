# CF-YOLO
# YOLOV11s-Enhanced

This project builds upon YOLOV11s with enhancements aimed at improving the model's accuracy and efficiency in small object detection.
The repository will provide model code, training and testing scripts, along with detailed introductions to the innovations.

## 🚀 Usage

To train the model:

1. Prepare your dataset and create a corresponding YAML configuration file (e.g., `data/my_dataset.yaml`).
2. Replace the path in `data/my_VisDrone.yaml` with your own dataset YAML file, or modify the training script to point to your config.
3. Run the training script:
   python train.py
Note: Make sure your dataset YAML file correctly specifies the paths to your training, validation, and class information.

🔧 Key Innovations：

✅ SBAM (Split-Block Attention Module)
Leverages a self-attention-guided split strategy to effectively integrate global and local contextual information, enhancing feature representation.

✅ CMFF (Cross-layer Multi-scale Feature Fusion Module)
Introduces cross-layer multi-scale feature interactions within the pyramid network, significantly improving localization and classification performance for small objects.

✅ MBD (Multi-Branch Downsampling Module)
Employs multi-branch downsampling to preserve high-information-density shallow features during spatial reduction, minimizing information loss.
