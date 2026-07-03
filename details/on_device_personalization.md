# On-Device Personalization for Consumer Electronics

[← Back to README](../README.md)

Running localized assistants (Mobile LLMs) on smartphones or laptops requires reducing their memory footprints.

## Application

Weight pruning combined with 4-bit quantization (BitsAndBytes) allows models to fit into standard mobile RAM, running locally without needing cloud API calls.

### Process Flow

```mermaid
graph TD
    A[User Query on Phone] --> B[Local Pruned & Quantized LLM]
    B --> C[Fast Local Processing]
    C --> D[Low-latency Response]
```

## Key Benefits

*   **Privacy:** Data never leaves the device.
*   **Offline Mode:** Works without internet access.
*   **Energy Efficiency:** Reduces processor load and saves battery.
