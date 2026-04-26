### Define Deeper Model Tooling >> Monitor Data-Ingestion -> Preprocessing -> Feature Transformation -> Modeling (Training-Dynamics) -> Postprocessing -> Inferencing || Monitor all the order-flow

import pandas as pd
import numpy as np
import torch

def trace_data_integrity(df_before: pd.DataFrame, df_after: pd.DataFrame):
    """Detects shifts in statistics and null counts between pipeline steps."""
    stats = {}
    for col in df_after.columns:
        if col in df_before.columns and pd.api.types.is_numeric_dtype(df_after[col]):
            mean_shift = abs(df_after[col].mean() - df_before[col].mean())
            null_growth = df_after[col].isnull().sum() - df_before[col].isnull().sum()
            
            stats[col] = {
                "mean_shift": mean_shift,
                "null_growth": null_growth,
                "drift_detected": mean_shift > (df_before[col].std() * 0.5) # Example threshold
            }
    return stats


def trace_tensor_logic(tensor: torch.Tensor, name: str, expected_semantics: list):
    """
    Checks if a tensor's shape matches the intended semantic labels.
    Example: expected_semantics=['Batch', 'Seq', 'Hidden']
    """
    actual_shape = list(tensor.shape)
    is_valid = len(actual_shape) == len(expected_semantics)
    
    return {
        "node": name,
        "shape": actual_shape,
        "semantics": expected_semantics,
        "valid_structure": is_valid,
        "metadata": f"Rank {len(actual_shape)} detected"
    }


def get_gradient_health(model: torch.nn.Module):
    """Traces the norm of gradients across layers to find bottlenecks."""
    grad_data = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_data[name] = {
                "norm": grad_norm,
                "status": "vanishing" if grad_norm < 1e-7 else "healthy"
            }
    return grad_data

def check_activation_sparsity(activation_tensor: torch.Tensor):
    """Calculates percentage of 'dead' (zero) neurons in a layer."""
    total_elements = activation_tensor.numel()
    zero_elements = torch.sum(activation_tensor == 0).item()
    sparsity = zero_elements / total_elements
    
    return {
        "sparsity_ratio": sparsity,
        "critical_dead": sparsity > 0.90 # Flag if 90% of layer is inactive
    }
    
def trace_loss_dynamics(loss_dict: dict):
    """Analyzes the contribution of different loss components."""
    total_loss = sum(loss_dict.values())
    analysis = {}
    
    for name, value in loss_dict.items():
        contribution = (value / total_loss) * 100
        analysis[name] = {
            "value": value,
            "contribution_pct": f"{contribution:.2f}%",
            "dominating": contribution > 80.0 # Flag if one loss ignores others
        }
    return analysis


def get_execution_context():
    """Captures the environment state for reproducibility debugging."""
    return {
        "python_version": sys.version.split()[0],
        "os": platform.system(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }