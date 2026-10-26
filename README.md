# FH4 AI Driver | Perception & Control (AMD ROCm Optimized)

This repository contains the core AI and Computer Vision engine for an autonomous driving system designed for *Forza Horizon 4*. Unlike standard AI projects, this implementation is specifically optimized for **AMD Radeon (RDNA 3)** hardware using the **AMD ROCm** stack, leveraging high-performance compute capabilities for real-time inference and training.

## 🚀 Hardware & Software Stack

A key highlight of this project is the seamless integration of Deep Learning models with AMD's ROCm platform, bypassing the standard CUDA-only workflow.

*   **Hardware:** **AMD Radeon RX 7900XT (RDNA 3 Architecture)**
*   **Compute Platform:** **AMD ROCm** (Radeon Open Compute)
*   **Deep Learning Framework:** **PyTorch (ROCm-optimized build)**
*   **Computer Vision:** OpenCV, BetterCam (Low-latency frame grabbing)
*   **System Integration:** Xbox Controller Emulation for real-time control loop

---

## 🧠 Core Modules

### 1. Data Acquisition & Multiprocessing (`makeTrainingData.py`)
*   **High-Throughput Pipeline:** Uses a multi-process producer-consumer architecture to decouple high-speed frame capture (90+ FPS) from heavy Disk I/O.
*   **Telemetry Sync:** Synchronizes visual frames with real-time game telemetry (speed, cornering force) into a structured dataset for supervised learning.
*   **Efficient Storage:** Automates large-scale screenshot collection and data organization using `multiprocessing.Queue`.

### 2. Deep Learning Training (`aiDeepLearning.py`)
*   **Mixed Precision Training:** Utilizes `torch.autocast` to exploit FP16/BF16 precision, significantly accelerating training on RDNA 3 hardware.
*   **Advanced Optimization:** Implements `CosineAnnealingLR` schedulers, non-blocking memory transfers, and `set_to_none=True` for efficient gradient management.
*   **Custom Loss Functions:** Employs a custom-weighted loss function (`LossFunc.py`) designed to prioritize critical driving behaviors (e.g., steering precision vs. throttle control).
*   **Automatic Safeguards:** Features a "Stagnation Shutdown" mechanism to stop training if the loss function converges or plateaus, preventing energy waste.

### 3. Real-Time Perception & Control (`evaluateInFH4.py`)
*   **Temporal Convolutional Neural Networks:** Processes sequences of frames (3D-CNN approach) to perceive vehicle momentum and environmental motion.
*   **Low-Latency Inference:** Optimized inference loop designed to run within the game's frame budget, enabling real-time control.
*   **Hardware-in-the-Loop (HIL):** Sends control signals (Steering, Gas, Brake) via a virtual Xbox controller, allowing the AI to interact directly with the game engine.

### 4. Reinforcement Learning (`reinforcement_learning.py`)
*   **Human-in-the-Loop RL:** Implements a reward-based optimization where human input provides a reward signal to refine the agent's policy.

---

## 📊 Performance Profiling (`testModelsProcessingTime.py`)

To ensure real-time viability, the system includes profiling tools to measure:
*   **Data Transformation Latency:** Time taken to convert raw camera frames into PyTorch tensors.
*   **Inference Latency:** Pure neural network execution time on the GPU.
*   **End-to-End Pipeline Latency:** Total time from frame capture to controller command.

---

## 🛠 Installation & Environment Setup

### Prerequisites
*   Python 3.10+
*   **AMD Radeon RX 7900XT**
*   **ROCm Drivers & Toolkit** installed on the host system.

### Setup Instructions
1. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
