# HCL-AL
HCL-AL is a hierarchical image-to-text retrieval framework for whole-body anatomical localization in automated radiology report generation. Using CLIP, the framework aligns lesion image patches with textual descriptions of anatomical locations in a shared embedding space, achieving millisecond-level precise lesion localization across nearly 400 anatomical sites.


# Releases
We have open-sourced the following components in addition to the core framework.
- Anatomical Localization Models (https://github.com/HerrBaum-YR/HCL-AL/releases/tag/v1.0.0-weights)
   - Coarse-Grained and Fine-Grained CLIP Pre-Trained Models for Lesion Localization

- Anatomical Localization Annotations (https://github.com/HerrBaum-YR/HCL-AL/releases/tag/v1.0.0-annotations)
   - The lesion annotations are sourced from the [AutoPET dataset] (https://autopet.grand-challenge.org/).
   - Some complex cases (number of lesions > 200) and false positives are excluded.
   - Each valid lesion was annotated with at least one precise anatomic localization tag (e.g., "Liver S1") following standardized radiological criteria.

# 🚀 Quick Start
- Install the HCL-AL framework
```
git clone https://github.com/HerrBaum-YR/HCL-AL
cd HCL-AL
# Install with pip
pip install -r requirements.txt
```

- Download pre-trained models and annotations in our releases

# 🧠 Model Inference
- Configure the inference configuration file by referring [./md_clip3d/config/inference_config.py]
- Perform inference
```
clip_apply -i ./inference_config.py
```
