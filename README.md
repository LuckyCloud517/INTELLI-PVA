# Informative Sample Annotation-based Contrastive Active Learning for cross-domain patient-ventilator asynchrony detection
---

## Abstract
Patient-ventilator asynchrony (PVA), a prevalent complication in mechanical ventilation, poses significant clinical challenges including prolonged ventilation duration, elevated mortality risk, and compromised patient outcomes. While artificial intelligence (AI) systems show promise for PVA detection, current methodologies face critical limitations in cross-institutional generalization due to substantial waveform pattern variations arising from the complex interplay between patient pathophysiology and ventilator operational parameters. These variations, further amplified by institution-specific ventilator control mechanisms, create substantial annotation bottlenecks in identifying representative waveform signatures from vast pools of morphologically similar cycles. To overcome these challenges, we present INTELLI-PVA - an INformaTivE contrastive active LearnIng framework for cross-domain PVA detection that synergizes two core innovations: 1) Random Augmentation-based Contrastive Learning (RACL) for robust feature representation, and 2) Multi-task Class-balanced Active Learning (MTCAL) for intelligent sample selection. This dual strategy enables efficient identification of high-information-density samples while maintaining pathological class balance, achieving domain adaptation with minimal annotation burden. This study establishes a paradigm-shifting approach for ventilator waveform analysis that effectively addresses key barriers to clinical AI implementation, with direct implications for improving critical care monitoring systems and ventilator management protocols.

## Getting Started
### part1: Supervised Training Mode
In the supervised training mode, the model is trained using labeled data, where each waveform is associated with a predefined label. The goal of this part is to learn a direct mapping from the input data to the corresponding output labels.

### part2: Random Augmentation-based Contrastive Learning (RACL)
Random Augmentation-based Contrastive Learning (RACL) aims to improve the model's ability to learn robust feature representations by augmenting the input data using random transformations. This contrastive learning framework is used to pre-train the model, where the goal is to learn feature embeddings that can differentiate between similar and dissimilar samples.

### part3: Contrastive Learning Framework Performance Validation
Linear Evaluation was employed as a fine-tuning strategy, wherein only the final linear classification layer of the pre-trained model is updated, while keeping the rest of the network frozen. This method is used to assess whether the features learned by the pre-trained model are linearly separable within the feature space.

### part4: Multi-task Class-balanced Active Learning (MTCAL)
In this part, a Multi-task Class-balanced Active Learning (MTCAL) framework is implemented. Leveraging the RACL-derived feature extractor, MTCAL iteratively identified and annotated the most informative samples from the target domain to achieve generalization across domains.

## Dependencies
---
```
conda create -n tf python=3.7

pip install tensorflow-gpu==2.5.0
pip install scikit-learn==1.0.2
pip install scikit-learn==1.18.0
pip install scipy==1.7.3
```

## Data
---
The data used in this study is not open access due to patient privacy and security concerns. 
