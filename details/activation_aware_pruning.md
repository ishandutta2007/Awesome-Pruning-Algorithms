# Activation-Aware Pruning (Wanda)

[← Back to README](../README.md)

Activation-aware pruning methods like Wanda (Weights and Activations) prune multi-billion parameter Large Language Models (LLMs) without requiring any retraining or backpropagation.

## How It Works

Wanda evaluates weight importance by multiplying the absolute value of the weight by the L2 norm of the input activations:

$$S_{ij} = |W_{ij}| \times \|X_j\|_2$$

Weights with the lowest score are pruned. This takes into account both the static weight value and the dynamic activations flowing through the network.

### Process Flow

```mermaid
graph TD
    A[Load Pre-trained LLM] --> B[Pass calibration data to collect activations X]
    B --> C[Compute importance score S_ij = |W_ij| * ||X_j||]
    C --> D[Prune lowest scoring weights per layer]
    D --> E[Immediate Zero-Shot Evaluation]
```

## Advantages & Limitations

*   **Pros:** Zero-shot; no training or fine-tuning required. Fast enough to prune 70B models in minutes.
*   **Cons:** Activation collection depends on calibration data distribution.
