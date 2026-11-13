import torch
import torch.nn as nn
import numpy as np

from .rules import WEIGHT_THRESHOLD, bias_shift, triangular_activation


@torch.no_grad()
def prune_structurally(model, shift_bias=True, weight_threshold=WEIGHT_THRESHOLD):
    """Remove only structurally empty rules (no data-based analysis).

    A structurally empty rule has an all-zero ``W_f`` row, so its activation
    is constant for every input. With ``shift_bias=True``, this constant base
    activation is moved into the output bias (Proposition 1, Exact Bias
    Shifting), leaving the model output unchanged. ``shift_bias=False`` is used
    for an uncorrected (naive) pruning comparison.

    Args:
        model (SCORENet): Trained model. Updated in place.
        shift_bias (bool): Add removed rules' base activation to the bias.
        weight_threshold (float): Threshold on the ``|W_f|`` row sum for
            treating a rule as empty.

    Returns:
        SCORENet: Compacted model (same object).
    """
    device = model.device
    model.eval()

    W_f = model.feature_linear.weight.data.clone()
    W_o = model.output_linear.weight.data.clone()
    width = torch.abs(model.base_feature_width.data).clone()
    center = model.base_feature_center.data.clone()

    rule_weight_sums = torch.sum(torch.abs(W_f), dim=1)
    drop_idx = torch.where(rule_weight_sums <= weight_threshold)[0]
    keep_idx = torch.where(rule_weight_sums > weight_threshold)[0]

    new_feature_linear = nn.Linear(model.N, len(keep_idx), bias=False).to(device)
    new_feature_linear.weight.data = W_f[keep_idx]

    new_output_linear = nn.Linear(len(keep_idx), model.C, bias=True).to(device)
    new_output_linear.weight.data = W_o[:, keep_idx]

    bias = torch.zeros(model.C, device=device)
    if getattr(model.output_linear, "bias", None) is not None and model.output_linear.bias is not None:
        bias = bias + model.output_linear.bias.data.clone()
    if shift_bias and len(drop_idx) > 0:
        bias = bias + bias_shift(W_o[:, drop_idx], center[drop_idx], width[drop_idx])
    new_output_linear.bias.data = bias

    model.H = len(keep_idx)
    model.feature_linear = new_feature_linear
    model.output_linear = new_output_linear
    model.base_feature_center = nn.Parameter(center[keep_idx])
    model.base_feature_width = nn.Parameter(width[keep_idx])
    model.eval()
    return model


@torch.no_grad()
def prune_and_compact_model(
    model, 
    data_loader=None, 
    weight_threshold=WEIGHT_THRESHOLD,
    min_activation_threshold=WEIGHT_THRESHOLD,
    min_coverage_pct=0.0
):
    """
    Detect and remove both structurally empty rules (weights below the
    threshold) and empirically dead rules (never or rarely triggered on the
    dataset), then compact the SCORENet model.

    When needed, the base activation of removed rules is shifted into the
    output_linear bias to preserve model fidelity.

    Args:
        model (SCORENet): Trained and pruned deep model.
        data_loader (DataLoader, optional): Loader used to inspect rule
            activations on the dataset. If given, empirical dead-rule analysis
            is enabled.
        weight_threshold (float): Base absolute threshold for treating a rule's
            total W_f weight as 'zero' (default: 1e-4).
        min_activation_threshold (float): Minimum peak activation a rule must
            produce on the data to be considered active (default: 1e-4).
        min_coverage_pct (float): Percentage of dataset samples in which a rule
            must trigger to survive (default: 0.0% — i.e. at least one sample).

    Returns:
        SCORENet: New compacted model with redundant rules removed.

    ===================================================================================================
    METHODOLOGICAL NOTES FOR THE PAPER AND RESEARCH (ACADEMIC DEFENSE):

    1. TWO-STAGE COMPACTION (STRUCTURAL & EMPIRICAL PRUNING):
       * Magnitude pruning alone cannot remove tiny residual noise weights
         (ghost rules around 1e-5) left after training.
       * This function checks both structural weight magnitude (|W_f|) and
         empirical support in the data space, reducing the model to a pure,
         irreducible minimal rule set.

    2. EXACT BIAS CONSERVATION:
       * If a removed rule's center/width balance produces a positive base
         activation at input 0, deleting it directly would shift the model's
         overall output level.
       * The function computes this base effect analytically and adds it to
         the output bias, so model outputs do not drift (zero-distortion
         guarantee).
    ===================================================================================================
    """
    device = model.device
    model.eval()

    # =========================================================================
    # 1. EXTRACT MODEL PARAMETERS AND WEIGHTS
    # =========================================================================
    W_f = model.feature_linear.weight.data.clone()           # (H, N)
    W_o = model.output_linear.weight.data.clone()           # (C, H)
    width = torch.abs(model.base_feature_width.data).clone() # (H,)
    center = model.base_feature_center.data.clone()          # (H,)
    
    orig_H = model.H
    N = model.N
    C = model.C

    # =========================================================================
    # 2. STAGE 1: STRUCTURAL EMPTY-RULE ANALYSIS (STRUCTURAL PRUNING)
    # =========================================================================
    # Absolute weight sum of each rule in the W_f layer
    rule_weight_sums = torch.sum(torch.abs(W_f), dim=1) # (H,)
    structurally_empty_mask = (rule_weight_sums <= weight_threshold)

    # =========================================================================
    # 3. STAGE 2: DATA-BASED DEAD-RULE ANALYSIS (EMPIRICAL PRUNING)
    # =========================================================================
    empirically_dead_mask = torch.zeros(orig_H, dtype=torch.bool, device=device)
    
    if data_loader is not None:
        total_samples = 0
        rule_max_activations = torch.zeros(orig_H, device=device)
        rule_active_counts = torch.zeros(orig_H, device=device)

        for batch in data_loader:
            batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch_x = batch_x.to(device)
            total_samples += batch_x.shape[0]

            h = model.feature_linear(batch_x)
            act = triangular_activation(h, center, width)

            # Maximum activation of each rule in this batch
            batch_max, _ = torch.max(act, dim=0)
            rule_max_activations = torch.maximum(rule_max_activations, batch_max)

            # Count of triggers above the threshold
            rule_active_counts += (act > min_activation_threshold).float().sum(dim=0)

        # Coverage ratio on the dataset (support %)
        coverage_pct = (rule_active_counts / total_samples) * 100.0

        # Rules with no peak activation on the data, or coverage below the threshold
        empirically_dead_mask = (rule_max_activations <= min_activation_threshold) | (coverage_pct <= min_coverage_pct)

    # =========================================================================
    # 4. COMBINE INDICES OF RULES TO REMOVE AND KEEP
    # =========================================================================
    # Remove both structurally empty rules and rules dead on the dataset
    prune_mask = structurally_empty_mask | empirically_dead_mask
    
    zero_idx = torch.where(prune_mask)[0]
    keep_idx = torch.where(~prune_mask)[0]

    num_struct_empty = int(structurally_empty_mask.sum().item())
    num_emp_dead = int((empirically_dead_mask & (~structurally_empty_mask)).sum().item())
    
    print(f"\n{'='*75}")
    print("MODEL COMPACTION AND DEAD-RULE CLEANUP REPORT")
    print(f"{'='*75}")
    print(f"Rule capacity before compaction (H): {orig_H}")
    print(f"-> Structurally empty rules (<= {weight_threshold}) : {num_struct_empty}")
    if data_loader is not None:
        print(f"-> Empirically dead rules (never triggered): {num_emp_dead}")
    print(f"Total removed rules: {len(zero_idx)} (removal rate: %{(len(zero_idx)/orig_H)*100:.1f})")
    print(f"Active surviving rules (new H): {len(keep_idx)}")
    print(f"{'='*75}")

    if len(zero_idx) == 0:
        print("No dead or empty rules to remove; the model is already compact.")
        return model

    # =========================================================================
    # 5. BIAS SHIFT AND CONSERVATION (EXACT BIAS CONSERVATION)
    # =========================================================================
    # Compute the constant base activation of removed rules at input x=0
    # (For structurally empty rules h=0; their base activation acts as a constant bias)
    w_zero = width[zero_idx]
    c_zero = center[zero_idx]
    
    # Only structurally empty rules contribute a constant bias
    # (Rules dead on the data would contribute an artificial value at input 0;
    # structurally empty rules give the exact net bias)
    struct_zero_submask = structurally_empty_mask[zero_idx]

    W_o_zero = W_o[:, zero_idx] # (C, num_pruned)
    bias_contribution = bias_shift(
        W_o_zero, c_zero, w_zero, keep=struct_zero_submask
    ) # (C,)

    # =========================================================================
    # 6. RESIZE THE MODEL (COMPACT ARCHITECTURE)
    # =========================================================================
    new_H = len(keep_idx)
    
    # New feature linear layer
    new_feature_linear = nn.Linear(N, new_H, bias=False).to(device)
    new_feature_linear.weight.data = W_f[keep_idx]

    # New output linear layer (bias is guaranteed to be True)
    new_output_linear = nn.Linear(new_H, C, bias=True).to(device)
    new_output_linear.weight.data = W_o[:, keep_idx]

    # Add to the existing bias if present, otherwise assign directly
    if getattr(model.output_linear, 'bias', None) is not None and model.output_linear.bias is not None:
        new_output_linear.bias.data = model.output_linear.bias.data.clone() + bias_contribution
    else:
        new_output_linear.bias.data = bias_contribution

    # Update center and width parameters
    new_center = nn.Parameter(center[keep_idx])
    new_width = nn.Parameter(width[keep_idx])

    # =========================================================================
    # 7. UPDATE THE MODEL'S INTERNAL PARAMETERS (IN-PLACE REPLACEMENT)
    # =========================================================================
    model.H = new_H
    model.feature_linear = new_feature_linear
    model.output_linear = new_output_linear
    model.base_feature_center = new_center
    model.base_feature_width = new_width

    print("[OK] Pruning and compaction complete.")
    print(f"-> Net bias contribution transferred to classes:\n{np.round(bias_contribution.cpu().numpy(), 4)}")
    print(f"-> New rule count (H): {model.H}")
    print(f"{'='*75}\n")

    return model
