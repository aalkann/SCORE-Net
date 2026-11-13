"""Figure helpers for the interpretable decision diagnostics.

Only :func:`plot_exact_actionable_recourse` is used by the released notebooks
(``03_pruning_figures.ipynb``). The six other plotting helpers that lived here
were written for exploratory analysis, are not referenced anywhere in the
repository, and are preserved in ``legacy/plot_utils_legacy.py`` so that the
history is not lost while the public API stays small and readable.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.analysis import *
from utils.rules import response_breakpoints, triangular_activation


def plot_exact_actionable_recourse(
    model, 
    single_record, 
    dataset_mins, 
    dataset_maxs, 
    target_class=1, 
    feature_names=None,
    top_k=None,
    figsize=None,
    title=None,
    xlabel=None,
    save_path=None,
    show_plot=True,
    return_actions=False,
    lang=None,
    immutable_features=None
):
    """
    Compute and plot every single-feature intervention that can flip the
    prediction, without simulating the model.

    The function locates the analytic zero-crossings of the piecewise-linear
    model and reports independent, one-feature-at-a-time recourse options.

    Args:
        model (SCORENet): Trained piecewise-linear model.
        single_record (Tensor/Array/List): Record whose prediction should change.
        dataset_mins (List/Array): Minimum value of each feature in the dataset.
        dataset_maxs (List/Array): Maximum value of each feature in the dataset.
        target_class (int): Index of the target class.
        feature_names (list, optional): Human-readable feature names.
        top_k (int, optional): Show only the K lowest-effort actions.
        figsize (tuple, optional): Matplotlib figure size.
        title (str, optional): Custom plot title.
        xlabel (str, optional): X-axis label.
        save_path (str, optional): Output path for the figure.
        show_plot (bool): Whether to call plt.show() (default: True).
        return_actions (bool): Whether to return valid_actions (default: False).
        immutable_features (list, optional): Features that cannot be changed.
            Given as indices (int) or name fragments (str, case-insensitive),
            e.g. ["age", "personal_status", "foreign_worker"]. No recourse
            action is generated for these features.

    Notes:
    * Exact recourse: the decision boundary is found by exact linear-algebra
      root-finding, not by gradient descent or genetic-search approximation.
    * User agency: each action changes exactly one feature, so the user can
      choose the cheapest independent alternative.
    """
    device = model.device
    
    # 1. Input type check and preparation
    if not isinstance(single_record, torch.Tensor):
        single_record = torch.tensor(single_record, dtype=torch.float32)
    if single_record.dim() == 1:
        single_record = single_record.unsqueeze(0)
        
    single_record = single_record.to(device)
    N = single_record.shape[1]
    
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(N)]

    # Resolve immutable features: index or name fragment.
    immutable_idx = set()
    if immutable_features is not None:
        for token in immutable_features:
            if isinstance(token, (int, np.integer)):
                immutable_idx.add(int(token))
                continue
            token_lower = str(token).lower()
            for j, name in enumerate(feature_names):
                if token_lower in str(name).lower():
                    immutable_idx.add(j)
        
    # Model parameters
    W_f = model.feature_linear.weight.data
    W_o = model.output_linear.weight.data
    C = model.base_feature_center.data
    W = torch.abs(model.base_feature_width.data)
    
    has_bias = getattr(model.output_linear, 'bias', None) is not None
    bias_o = model.output_linear.bias.data if has_bias else 0
    
    # Current prediction
    orig_logits = model(single_record)[0]
    orig_class = orig_logits.argmax().item()
    
    print(f"\n{'='*80}")
    print(f"ANALYTIC EXACT SOLUTION: Current class [{orig_class}] ---> Target class [{target_class}]")
    print(f"{'='*80}")
    
    if orig_class == target_class:
        print("This record is already in the target class; no plot is needed.")
        if return_actions:
            return []
        return
        
    # 2. Find valid alternatives (analytic computation)
    valid_actions = []
    
    for feature_idx in range(N):
        if feature_idx in immutable_idx:
            continue  # Immutable features are not recourse candidates.
        m = W_f[:, feature_idx] 
        if torch.all(m == 0): continue # Skip features not connected to any rule
            
        b = model.feature_linear(single_record)[0] - (m * single_record[0, feature_idx])
        
        orig_val = single_record[0, feature_idx].item()
        f_min, f_max = dataset_mins[feature_idx].item() if isinstance(dataset_mins[feature_idx], torch.Tensor) else dataset_mins[feature_idx], dataset_maxs[feature_idx].item() if isinstance(dataset_maxs[feature_idx], torch.Tensor) else dataset_maxs[feature_idx]

        # Breakpoints come from a single source (utils.rules), matching the
        # paper's breakpoint equation, so the two cannot diverge. A superscript
        # marks the rule's slice edge in z-space; when W_f,hj < 0, v_minus may
        # exceed v_plus, and ``response_breakpoints`` handles the ordering.
        sorted_points = response_breakpoints(
            m,
            C,
            W,
            b,
            lower=float(f_min),
            upper=float(f_max),
            extra=[orig_val],
            decimals=5,
            include_bounds=False,
        )
        K = len(sorted_points)
        test_v = sorted_points.unsqueeze(1) 
        
        h_K = m.unsqueeze(0) * test_v + b.unsqueeze(0) 
        act_K = triangular_activation(h_K, C, W)
        logits_K = torch.matmul(act_K, W_o.T) + bias_o 
        
        # Margin needed for the target class to win
        diff_K = logits_K[:, target_class] - logits_K[:, orig_class] 
        
        def _margin_at(value):
            """Return the logit margin for the target class at a feature value."""
            probe = single_record.clone()
            probe[0, feature_idx] = value
            with torch.no_grad():
                probe_logits = model(probe)[0]
            return float(probe_logits[target_class] - probe_logits[orig_class])

        best_v_cross = None
        min_dist = float('inf')

        for i in range(K - 1):
            y1 = diff_K[i].item()
            y2 = diff_K[i+1].item()
            v1 = sorted_points[i].item()
            v2 = sorted_points[i+1].item()

            # Crossing point: geometric root of the interval where the decision flips
            if (y1 <= 0 and y2 > 0) or (y1 > 0 and y2 <= 0):
                v_cross = v1 - y1 * (v2 - v1) / (y2 - y1)

                # Check that the crossing stays inside the feature bounds
                if not (f_min <= v_cross <= f_max): continue

                # At the root the model is tied (margin = 0). For a sufficient
                # intervention, step a small safety margin into the target-class
                # side and verify the margin.
                direction = 1.0 if v2 > v1 else -1.0
                if y2 <= 0:
                    direction = -direction
                limit = v2 if direction > 0 else v1
                delta = max(1e-6, 1e-3 * abs(v2 - v1))
                target_value = None
                for _ in range(60):
                    candidate = v_cross + direction * delta
                    if (direction > 0 and candidate > limit) or (
                        direction < 0 and candidate < limit
                    ):
                        break
                    if _margin_at(candidate) > 0:
                        target_value = candidate
                        break
                    delta *= 2.0

                if target_value is None:
                    continue

                dist = abs(target_value - orig_val)
                # Keep the candidate closest to the original value (lowest effort)
                if dist < min_dist and dist > 1e-4:
                    min_dist = dist
                    best_v_cross = target_value
                    
        # Add a valid crossing for this feature, if any
        if best_v_cross is not None:
            diff = best_v_cross - orig_val
            valid_actions.append({
                'feature': feature_names[feature_idx],
                'orig': orig_val,
                'target': best_v_cross,
                'diff': diff,
                'abs_diff': abs(diff)
            })

    # 3. Plotting (Matplotlib)
    if not valid_actions:
        print("The decision cannot be flipped by changing only one feature.")
        print("At least two features must change together to reach the target class.")
        if return_actions:
            return []
        return
        
    # Sort from lowest effort to highest effort
    valid_actions.sort(key=lambda x: x['abs_diff'])
    displayed_actions = valid_actions[:top_k] if top_k is not None else valid_actions
    
    # Prepare plot data
    labels = [item['feature'] for item in displayed_actions]
    diffs = [item['diff'] for item in displayed_actions]
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in diffs] # Increase green, decrease red
    
    calc_figsize = figsize if figsize is not None else (12, max(6, len(labels) * 0.8))
    fig, ax = plt.subplots(figsize=calc_figsize)
    
    # Horizontal bar chart
    bars = ax.barh(labels, diffs, color=colors, edgecolor='black', alpha=0.8)
    
    # Draw the zero axis
    ax.axvline(0, color='black', linewidth=1.5)
    
    # Annotate each bar with its original and new value.
    # Text is placed on the right (x >= 0) also for negative bars so that the
    # y-axis feature labels stay visible.
    max_abs_diff = max(abs(d) for d in diffs) if diffs else 1.0
    offset = max_abs_diff * 0.03
    
    for bar, action in zip(bars, displayed_actions):
        width = bar.get_width()
        x_pos = max(0.0, width) + offset

        annotation = f"Current: {action['orig']:+.2f} \u2192 Target: {action['target']:+.2f} (\u0394 = {action['diff']:+.2f})"
            
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, annotation, 
                va='center', ha='left', fontsize=9.0, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1))
                
    # Expand axis limits so annotations are not clipped
    min_val = min(min(diffs) * 1.25, -0.05) if (diffs and min(diffs) < 0) else -0.05
    max_val = max(max(diffs), 0.0) + max_abs_diff * 1.15 if diffs else 1.0
    ax.set_xlim(min_val, max_val)

    # Axis and title settings
    ax.set_xlabel(xlabel if xlabel is not None else "Required Change (Delta)", fontsize=11, fontweight='bold')
    if title is not None:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    else:
        ax.set_title(f"Independent Alternative Actions Sufficient for Class {target_class}\n(Sorted from Lowest to Highest Effort)",
                     fontsize=13, fontweight='bold', pad=15)
    
    # Remove top and right spines for a clean academic look
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.gca().invert_yaxis() # Put the best (smallest) change on top
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
        if save_path.endswith('.pdf'):
            png_fallback = save_path[:-4] + '.png'
            fig.savefig(png_fallback, bbox_inches='tight', dpi=300)
            print(f"PNG copy saved: {png_fallback}")
        elif save_path.endswith('.png'):
            pdf_fallback = save_path[:-4] + '.pdf'
            fig.savefig(pdf_fallback, bbox_inches='tight')
            print(f"PDF copy saved: {pdf_fallback}")
        
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
        
    if return_actions:
        return valid_actions
