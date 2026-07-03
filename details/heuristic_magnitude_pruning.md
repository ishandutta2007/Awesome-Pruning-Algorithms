# Heuristic Magnitude-Based Pruning

[← Back to README](../README.md)

Magnitude-based pruning is a simple, heuristic-driven approach popularized by Song Han et al. in 2015. It focuses on removing weights whose absolute values are below a specified threshold.

## How It Works

This method assumes that smaller weights contribute less to the output of the network. It follows an iterative process of training, pruning, and fine-tuning to recover accuracy.

### Process Flow

```mermaid
graph TD
    A[Train Dense Model] --> B[Evaluate Weight Absolute Values |w|]
    B --> C[Mask weights where |w| < threshold to 0]
    C --> D[Fine-tune/Retrain Sparse Model]
    D --> E{Meets Accuracy & Sparsity Target?}
    E -- No --> B
    E -- Yes --> F[Deploy Sparse Model]
```

## Advantages & Limitations

*   **Pros:** Highly scalable, easy to implement, and requires no expensive second-order gradient calculations.
*   **Cons:** Leads to unstructured sparsity, which standard hardware (GPUs/CPUs) cannot easily accelerate without custom sparse kernels.
