# ✂️ Awesome-Pruning-Algorithms

<!-- SEO Meta Tags -->
<meta name="description" content="A curated list of awesome neural network pruning algorithms, hardware-fused semi-structured blocks, magnitude-based sparsification, and zero-shot post-training LLM compression methods like Wanda." />
<meta name="keywords" content="Neural Network Pruning, Model Compression, LLM Sparsification, Structured Pruning, Unstructured Pruning, 2:4 Sparsity, Wanda, Optimal Brain Damage, Lottery Ticket Hypothesis, Deep Learning Optimization" />

<p align="center">
  <img src="assets/banner.svg" alt="Awesome Pruning Algorithms Banner" width="100%">
</p>

<p align="center">
  <a href="https://github.com/ishandutta2007/Awesome-Awesome-Awesome"><img src="https://img.shields.io/badge/Awesome-%E2%9C%94-blueviolet?style=flat-square&logo=github" alt="Awesome"/></a><a href="https://discord.gg/jc4xtF58Ve"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord" /></a> <a href="https://github.com/ishandutta2007/Awesome-Pruning-Algorithms/stargazers"><img src="https://img.shields.io/github/stars/ishandutta2007/Awesome-Pruning-Algorithms" alt="GitHub Stars"/></a> <a href="https://github.com/ishandutta2007/Awesome-Pruning-Algorithms/issues"><img src="https://img.shields.io/github/issues/ishandutta2007/Awesome-Pruning-Algorithms" alt="GitHub Issues"/></a> <a href="https://github.com/ishandutta2007/Awesome-Pruning-Algorithms/blob/main/LICENSE"><img src="https://img.shields.io/github/license/ishandutta2007/Awesome-Pruning-Algorithms" alt="License"/></a> <a href="https://github.com/ishandutta2007"><img alt="GitHub followers" src="https://img.shields.io/github/followers/ishandutta2007?label=Follow" /></a>
</p>

## 🧠 Pruning Algorithms in AI: History, Progression, Variants, & Applications

**Pruning Algorithms** represent a highly specialized family of hardware-aware model compression and optimization frameworks designed to eliminate redundant, non-essential parameters or architectural blocks from a neural network [INDEX: 16]. The primary objective of pruning is to reduce the computational complexity (FLOPs), memory footprint, and storage size of deep learning models without sacrificing their baseline cognitive reasoning capabilities [INDEX: 16]. 

By strategically setting low-impact network parameters to an absolute value of zero, pruning transforms dense tensor matrices into **sparse matrices** [INDEX: 16]. This allows specialized hardware engines and sparse compilers to bypass unnecessary zero-value floating-point operations entirely, accelerating inference throughput and enabling multi-billion parameter foundation architectures to deploy efficiently within resource-constrained edge computing environments [INDEX: 16].

---

## ⏳ 1. The Macro Chronological Evolution

The technical optimization of network sparsification has transitioned from analytic second-order derivative estimation to heuristic magnitude sorting, moving toward hardware-fused semi-structured blocks and dynamic, activation-aware pruning routines.

```mermaid
flowchart LR
    A["Analytical Second-Order (OBD, 1989)<br/>(Hessian-Based Pruning)"]
    --> B["Magnitude-Based Pruning (2015)<br/>(Unstructured Weight Magnitude Thresholding)"]
    --> C["Semi-Structured 2:4 Pruning (2020)<br/>(Hardware-Friendly Tensor Core Sparsity)"]
    --> D["Activation-Aware Pruning (Wanda, 2024+)<br/>(Zero-Shot Post-Training LLM Compression)"]
```

| Era / Paradigm | First Used Year | First Used Paper | Details |
| :--- | :--- | :--- | :--- |
| [**The Analytical Second-Order Era (Optimal Brain Damage)**](details/optimal_brain_damage.md) | 1989 | [Optimal Brain Damage](https://papers.nips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) | **Concept:** The theoretical genesis of network sparsification [INDEX: 16]. It modeled pruning as an explicit optimization task, utilizing a second-order Taylor expansion to calculate the "salience" (importance) of each weight parameter based on the diagonal elements of the Hessian matrix [INDEX: 16]. Weights demonstrating the lowest impact on final loss were permanently deleted [INDEX: 16].<br><br>**Limitation:** Computationally prohibitive at scale [INDEX: 16]. Computing or even approximating the explicit Hessian matrix over a network introduces an intolerable $O(N^2)$ tracking overhead, rendering it completely unscalable for deep networks containing millions or billions of parameters [INDEX: 16]. |
| [**The Heuristic Magnitude-Based Era**](details/heuristic_magnitude_pruning.md) | 2015 | [Learning both weights and connections for efficient neural network](https://arxiv.org/abs/1506.02626) | **Concept:** Sparked the modern open-source model compression boom [INDEX: 16]. Han et al. bypassed heavy second-order calculus by introducing a highly scalable iterative pipeline: train a dense network, drop all parameters whose absolute value falls below a specific threshold ($\vert w\vert < \epsilon$), and fine-tune the remaining sparse connections to patch up accuracy gaps [INDEX: 16].<br><br>**Limitation:** Produced unstructured sparsity [INDEX: 16]. While mathematically elegant, scattering zeros randomly across weight matrices creates irregular, fragmented memory access patterns that standard GPU hardware cannot translate into real-world wall-clock speedups [INDEX: 16]. |
| [**The Hardware-Fused & Semi-Structured Era**](details/semi_structured_pruning.md) | 2021 | [Accelerating Sparse Deep Neural Networks on NVIDIA Ampere GPUs via 2:4 Semi-Structured Patterns](https://arxiv.org/abs/2104.08378) | **Concept:** Bridges the gap between algorithmic sparsity and physical silicon layout [INDEX: 16]. Popularized by architectures like NVIDIA Ampere and Hopper GPUs, it enforces strict **N:M semi-structured sparsity** templates [INDEX: 16]. The pruning compiler scans weight tensors, forcing exactly $N$ zeros inside every $M$ contiguous parameter block (most notably **2:4 Sparsity**), activating hardwired acceleration loops inside modern Tensor Cores to deliver a $2\times$ processing speedup [INDEX: 16]. |
| [**The Activation-Aware Post-Training LLM Era**](details/activation_aware_pruning.md) | 2023 | [A Simple and Effective Pruning Method for Large Language Models](https://arxiv.org/abs/2306.11695) | **Concept:** The current modern state-of-the-art frontier standard engineered specifically to prune multi-billion parameter foundation transformers without costly retraining. Frameworks like **Wanda (Pruning Weights and Activations)** move past looking at weights in isolation.<br><br>**Significance:** It evaluates the product of weight magnitudes and input activation norms ($\vert w_{ij}\vert \times \|X_j\|_2$) dynamically. This allows engineers to prune massive networks zero-shot in a single forward pass without running fine-tuning epochs, protecting complex reasoning capabilities perfectly. |

---

## 🧩 2. Core Granularity & Structural Variants

Pruning methodologies are strictly categorized based on the geometric patterns and physical layouts of the parameter elements targeted for elimination [INDEX: 16].

| Variant | First Used Year | First Used Paper | Details |
| :--- | :--- | :--- | :--- |
| [**A. Unstructured Pruning (Element-Wise Deletion)**](details/unstructured_pruning.md) | 1989 | [Optimal Brain Damage](https://papers.nips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) | **Mechanism:** Targets individual, isolated weight coordinates anywhere within a tensor matrix completely independent of neighboring values [INDEX: 16].<br><br>**Pros:** Achieves exceptionally high compression ratios (up to 90–95% parameters removed) with minimal degradation in final model reasoning [INDEX: 16].<br><br>**Cons:** Requires highly specialized sparse software kernels; standard hardware treats it as irregular memory overhead, resulting in zero execution speedup [INDEX: 16]. |
| [**B. Structured Pruning (Channel / Filter Deletion)**](details/structured_pruning.md) | 2015 | [Structured Pruning of Deep Convolutional Neural Networks](https://arxiv.org/abs/1512.08571) | **Mechanism:** Eliminates entire cohesive architectural blocks wholesale—such as an entire convolutional filter matrix, an attention head block, or an entire hidden layer row dimension [INDEX: 16].<br><br>**Pros:** Natively compatible with any standard, off-the-shelf CPU or GPU hardware accelerator without requiring custom compilers [INDEX: 16]. The pruned matrix simply shrinks into a smaller, dense matrix, instantly compressing FLOPs and latency [INDEX: 16]. |
| [**C. Semi-Structured Pruning (N:M Sparsity)**](details/nm_sparsity.md) | 2021 | [Accelerating Sparse Deep Neural Networks on NVIDIA Ampere GPUs via 2:4 Semi-Structured Patterns](https://arxiv.org/abs/2104.08378) | **Mechanism:** A hardware-software co-design standard [INDEX: 16]. It enforces a strict bounded ratio template—most notably **2:4 Sparsity**, where out of every 4 contiguous elements along a matrix row, exactly 2 elements must be compressed to zero [INDEX: 16].<br><br>**Pros:** Capitalizes on native, hardwired structural acceleration inside modern GPU Tensor Cores, delivering an instantaneous $2\times$ mathematical processing speedup [INDEX: 16]. |

---

## ⚙️ 3. Training Dynamics & Pruning Schedules

Depending on where the parameter elimination step intersects with the model optimization timeline, pruning schedules follow distinct operational pipelines [INDEX: 16].

| Schedule / Pipeline | First Used Year | First Used Paper | Details |
| :--- | :--- | :--- | :--- |
| [**Post-Training Pruning (PTP / PT-Sparsification)**](details/post_training_pruning.md) | 1989 | [Optimal Brain Damage](https://papers.nips.cc/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html) | **Pipeline:** Takes a fully trained, converged model and applies a one-shot pruning pass based on weight magnitudes or activation-aware metrics, followed by an immediate downstream fine-tuning sprint over a small data subset to patch up accuracy gaps [INDEX: 16]. |
| [**Pruning-Aware Training (PAT / Sparse Training)**](details/pruning_aware_training.md) | 2015 | [Learning both weights and connections for efficient neural network](https://arxiv.org/abs/1506.02626) | **Pipeline:** Integrates parameter elimination straight into the active backpropagation loop [INDEX: 16]. During the forward pass, weights are masked to zero based on dynamic metrics [INDEX: 16]. During the backward pass, gradients are calculated over the unmasked parameters, allowing the network to continuously adapt its remaining connections [INDEX: 16]. |
| [**The Lottery Ticket Hypothesis Pipeline**](details/lottery_ticket_hypothesis.md) | 2018 | [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) | **Pipeline:** Proved by Frankle & Carbin (2018) that dense networks contain sub-networks ("winning tickets") that can match the accuracy of the original model when trained in isolation from scratch [INDEX: 16]. The pipeline prunes a model, resets the remaining weights back to their exact *initialization state step zero*, and re-trains the sparse mask cleanly [INDEX: 16]. |

---

## 💾 4. Production Engineering Challenges & Hardware Solutions

Deploying sparse networks across commercial production nodes requires balancing algorithmic compression metrics against underlying silicon memory buses [INDEX: 16].

| Challenge / Solution | First Used Year | First Used Paper | Details |
| :--- | :--- | :--- | :--- |
| [**The Sparse Memory-Access Latency Penalty**](details/sparse_memory_latency.md) | 2016 | [EIE: Efficient Inference Engine on Compressed Deep Neural Network](https://arxiv.org/abs/1602.01527) | **The Problem:** Storing an unstructured sparse matrix requires saving auxiliary indexing arrays (like Compressed Sparse Row - CSR tables) to track where the non-zero parameters reside [INDEX: 16]. Reading these fragmented index files from slow High Bandwidth Memory (HBM) saturates the memory bus, frequently making unstructured sparse models *slower* than dense models in production [INDEX: 16].<br><br>**Mitigation:** Transitioning entirely to **Block-Wise or 2:4 Semi-Structured Pruning structures**, allowing hardware memory decoders to stream parameter rows using predictable, contiguous caching bursts [INDEX: 16]. |
| [**The Capacity-Starved Divergence Boundary**](details/capacity_starved_divergence.md) | 2023 | [LLM-Pruner: On-Device Large Language Model Compression](https://arxiv.org/abs/2305.11627) | **The Problem:** When scaling pruning past a specific threshold (e.g., trying to eliminate $70\%$ of parameters from a compact 3B model), the network experiences sudden, irreversible capability collapse [INDEX: 16]. It drops complex reasoning features (like software coding syntax or mathematical verification) first, rendering the compressed model useless for advanced tasks [INDEX: 16].<br><br>**Mitigation:** Transitioning away from uniform, flat pruning across all layers toward **Layer-Wise Sensitivity Allocation (such as Wanda or Taylor-approximation metrics)**, which shields critical attention gates and early projection matrices while heavily pruning wide, redundant hidden MLP columns [INDEX: 15, 16]. |

---

## 🚀 5. Frontier Real-World AI Industrial Applications

| Application | First Used Year | First Used Paper / Source | Details |
| :--- | :--- | :--- | :--- |
| [**Low-Latency Enterprise LLM Inference Serving**](details/low_latency_llm_serving.md) | 2023 | [TensorRT-LLM Release / NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | **Application:** Optimizes cloud serving frameworks for high-volume concurrent streams [INDEX: 16]. By applying 2:4 semi-structured pruning to large open-weights foundation models, enterprise servers activate native sparse tensor cores, cutting **Time-to-First-Token (TTFT)** metrics and doubling query throughput per node [INDEX: 16]. |
| [**Autonomous Vehicle Microcontroller Compression**](details/autonomous_vehicle_compression.md) | 2020 | [MCUNet: TinyDL on IoT Devices](https://arxiv.org/abs/2007.10319) | **Application:** Compresses deep convolutional and vision-language stacks to fit within the highly restricted VRAM and thermal power caps of vehicle chipsets [INDEX: 1, 16]. Structured filter pruning scales down the computer vision perception graph, letting edge cards run object classification and depth segmentation maps in real time [INDEX: 1, 16]. |
| [**On-Device Personalization for Consumer Electronics**](details/on_device_personalization.md) | 2024 | [MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use](https://arxiv.org/abs/2402.14905) | **Application:** Running localized assistants on mobile devices or consumer laptops [INDEX: 16]. Weight pruning, layered alongside **BitsAndBytes 4-bit Quantization**, compresses model footprints down into standard system RAM boundaries, preserving device battery life during offline conversational sampling loops [INDEX: 16]. |

---

## 📚 References
1. LeCun, Y., Denker, J., & Solla, S. (1989). Optimal brain damage. *Advances in Neural Information Processing Systems (NeurIPS)*, 2 [INDEX: 16].
2. Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural network. *Advances in Neural Information Processing Systems (NeurIPS)*, 28, 1135-1143 [INDEX: 16].
3. Frankle, J., & Carbin, M. (2018). The lottery ticket hypothesis: Finding sparse, trainable neural networks. *International Conference on Learning Representations (ICLR)* [INDEX: 16].
4. Hoefler, T., et al. (2021). Sparsity in deep learning: Pruning and growth for efficient inference and training. *Journal of Machine Learning Research*, 22(241), 1-124.
5. Mishra, A., et al. (2021). Accelerating sparse deep neural networks on NVIDIA Ampere GPUs via 2:4 semi-structured patterns. *NVIDIA whitepaper architecture manifestos* [INDEX: 16].
6. Sun, M., et al. (2024). A simple and effective pruning method for large language models. *International Conference on Learning Representations (ICLR)* [INDEX: 16].

##  Star History
<div align="center">
<a href="https://www.star-history.com/?repos=ishandutta2007%2FAwesome-Pruning-Algorithms&type=date&legend=bottom-right">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Pruning-Algorithms&type=date&theme=dark&legend=bottom-right" />
<source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Pruning-Algorithms&type=date&legend=bottom-right" />
<img alt="Star History Chart" src="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Pruning-Algorithms&type=date&legend=bottom-right" />
</picture>
</a>
</div>

