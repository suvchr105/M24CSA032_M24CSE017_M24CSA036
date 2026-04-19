# AVQACL Novelty: Enhanced Audio-Visual Question Answering
<img width="1372" height="784" alt="T-AVQCL architecture" src="https://github.com/user-attachments/assets/5552ada9-9693-4e74-ac83-6131a3b8b672" />

This directory contains the **Novelty Implementation** of the Audio-Visual Question Answering Continual Learning (AVQACL) framework. This version introduces several state-of-the-art enhancements designed to improve cross-modal representation learning, increase parameter efficiency, and mitigate catastrophic forgetting through representative memory selection.

## 🚀 Key Innovations

### 1. Transformer-based Backbones
Unlike the baseline which uses traditional CNNs and RNNs, the Novelty version leverages specialized transformer architectures for all modalities:
- **Visual**: [Swin Transformer](https://github.com/microsoft/Swin-Transformer) (`swin_tiny_patch4_window7_224`) for hierarchical spatial modeling.
- **Audio**: [Audio Spectrogram Transformer (AST)](https://github.com/YuanGongnd/ast) for global temporal-frequency modeling of audio signals.
- **Question/Text**: [DistilBERT](https://huggingface.co/distilbert-base-uncased) for efficient and robust language understanding.

### 2. Parameter-Efficient Fine-Tuning (Adapters)
To handle the scale of transformer models in a continual learning setting, we implement **Bottleneck Adapters**.
- **Efficiency**: The pre-trained backbones are frozen, and only lightweight adapter layers (Linear → ReLU → Linear) are trained.
- **Stability**: Using adapters prevents the degradation of pre-trained weights during incremental steps, providing a more stable foundation for learning new tasks.

### 3. Cross-Modal Contrastive Learning
To ensure that audio and visual features are well-aligned before fusion, we introduce an **InfoNCE Contrastive Loss** during the training process.
- **Mechanism**: Projects audio and visual sequence features into a joint latent space.
- **Benefit**: Forces the model to learn a shared representation, making the subsequent fusion step more effective for answering modal-dependent questions.

### 4. Prototype-based Memory Selection
Standard rehearsal methods often use random or simple herding selection. Our Novelty implementation uses **Prototype Selection via K-Means Clustering**:
- **Mechanism**: After each task, we extract features for all samples and perform K-Means clustering in the fused feature space.
- **Strategy**: The samples closest to the centroids (the "prototypes") are selected for the permanent memory buffer.
- **Outcome**: Ensures the memory buffer contains the most representative and diverse samples of each class, leading to significantly lower forgetting rates.

---

## 🏗️ Architecture Overview

The model follows a structured pipeline:
1. **Feature Extraction**: Parallel processing via Swin-V, AST, and DistilBERT.
2. **Adapter Injection**: Task-specific adapters modulate the backbone features.
3. **Cross-Attention Fusion**: Question features act as the Query (Q) to attend over Visual and Audio Key/Values (K, V).
4. **Contrastive Head**: Simultaneous modal alignment using InfoNCE loss.
5. **Incremental Head**: Dynamic classifier growth to accommodate new answer classes.

---

## 📊 Loss Formulation

The total training objective is defined as:
$$L_{total} = L_{cls} + \lambda L_{contrast}$$

Where:
- $L_{cls}$: Cross-Entropy loss for the question-answering task.
- $L_{contrast}$: InfoNCE loss for cross-modal alignment.
- $\lambda$: Weighting factor for the contrastive objective (default: 0.1).

---

## 🛠️ Usage

To initiate training with the novelty features, use the `train_incremental_novelty.py` script:

```bash
python train_incremental_novelty.py \
    --train_batch_size 16 \
    --lr 1e-4 \
    --use_adapters True \
    --use_transformer_encoders True \
    --use_contrastive_loss True \
    --use_prototype_memory True \
    --memory_size 700
```

### Key Arguments:
- `--use_adapters`: Enables the bottleneck adapter layers.
- `--use_transformer_encoders`: Toggles the use of Swin, AST, and DistilBERT.
- `--use_contrastive_loss`: Enables the cross-modal InfoNCE alignment.
- `--use_prototype_memory`: Switches from random to prototype-based memory selection.

---

## 📁 Directory Structure

- `audio_visual_model_novelty.py`: Core architecture with transformer encoders and adapters.
- `memory_novelty.py`: Implementation of K-Means prototype selection.
- `dataloader_novelty.py`: Specialized loaders for transformer-compatible inputs (raw pixels/audio/IDs).
- `train_incremental_novelty.py`: Main training loop for the novelty pipeline.
