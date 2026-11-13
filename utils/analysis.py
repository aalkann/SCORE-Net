import torch
import numpy as np
import pandas as pd

from .rules import (
    EPS,
    WEIGHT_THRESHOLD,
    integer_weights,
    per_rule_scales,
    quantize_projection,
    quantize_projection_uniform,
    triangular_activation,
)


############################################################################################
############################################################################################
############################################################################################

@torch.no_grad()
def calculate_global_feature_importance(model, data_loader=None, feature_names=None):
    """
    Compute the Analytical Global Feature Importance for the SCORENet architecture.

    The score combines three terms: the within-rule feature weights (|W_f|), the overall
    decision strength of a rule over the output classes (||W_o||_1), and the rule's actual
    representational power in input space (Empirical Coverage). Their product gives each
    feature's total contribution to the model decisions.

    Mathematical formulation:
        Importance(f) = sum_h ( |W_f[h, f]| * ||W_o[:, h]||_1 * Coverage_h )

    Args:
        model (SCORENet): Trained piecewise-linear deep learning model.
        data_loader (DataLoader, optional): Data loader used to estimate rule activation
            coverage (Coverage) empirically over the real data distribution. If not given,
            the geometric rule width (Width) is used instead.
        feature_names (list of str, optional): Feature names used in the report.

    Returns:
        pd.DataFrame: Table with raw importance scores ('Importance_Score') and percentage
        decision shares ('Importance_Pct'), sorted from highest to lowest.

    ===================================================================================================
    METHODOLOGICAL NOTES FOR THE PAPER:
    
    1. INTRINSIC EXPLAINABILITY:
       * Traditional methods such as SHAP or Permutation Importance run the model thousands of
         times and produce a statistical approximation.
       * This function instead solves the model's internal weight tensors and rule geometry
         directly and algebraically. It has no approximation error (fidelity loss) and is
         deterministic.

    2. HYBRID COVERAGE (COVERAGE-SENSITIVITY BALANCE):
       * Using only the |W_f| * |W_o| product could make "dead rules" that never fire on the data
         look very important.
       * Including the expected rule activation (E[a_h(x)]) in the equation discounts narrow or
         overfitted rules and highlights only general rules that truly represent the data.
    ===================================================================================================
    """
    device = model.device
    model.eval()

    # =========================================================================
    # 1. EXTRACT MODEL PARAMETERS AND MATRICES
    # =========================================================================
    # W_f: linear contribution weights from input features to rule centers (H, N)
    W_f = model.feature_linear.weight.data.abs()
    
    # W_o: logit score weights a rule distributes to the output classes (C, H)
    W_o = model.output_linear.weight.data.abs()
    
    # width: radius defining how large a region of input space the rule covers (H,)
    width = torch.abs(model.base_feature_width.data)
    
    # center: peak of rule activation (a_h = 1.0) (H,)
    center = model.base_feature_center.data

    H, N = W_f.shape
    
    # Total decision effect of each rule (H) over all classes (L1 norm): (H,)
    rule_output_importance = W_o.sum(dim=0)

    # =========================================================================
    # 2. COMPUTE RULE COVERAGE
    # =========================================================================
    # A rule may have a large output weight, but its contribution to the overall
    # decision is zero if it never fires on the data.
    if data_loader is not None:
        # A) EMPIRICAL DATA COVERAGE (data-dependent coverage)
        total_samples = 0
        total_activation = torch.zeros(H, device=device)

        for batch in data_loader:
            batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch_x = batch_x.to(device)
            total_samples += batch_x.shape[0]

            # Linear combination input of the rule
            h = model.feature_linear(batch_x)
            
            # Tent/triangular activation strength of the rule, in [0, 1]
            act = triangular_activation(h, center, width)
            
            # Accumulate activations over the whole batch
            total_activation += act.sum(dim=0)

        # Mean expected rule activation over the dataset E[a_h(x)]: (H,)
        rule_coverage = total_activation / total_samples
    else:
        # B) GEOMETRIC VOLUME COVERAGE (data-independent coverage)
        # Without a data loader, theoretical coverage is proportional to the rule width
        rule_coverage = width / (width.max() + EPS)

    # =========================================================================
    # 3. DERIVE THE EFFECTIVE RULE WEIGHT
    # =========================================================================
    # Effective weight = (rule output decision strength) * (actual firing strength on data)
    # This step drives the score of high-weight but "dead/narrow" rules to zero.
    effective_rule_weight = rule_output_importance * rule_coverage  # Shape: (H,)

    # =========================================================================
    # 4. COMPUTE GLOBAL IMPORTANCE SCORES PER FEATURE
    # =========================================================================
    # Feature impact matrix: each rule's feature share times the rule's effective weight
    # (H, N) * (H, 1) -> (H, N)
    feature_impact_matrix = W_f * effective_rule_weight.unsqueeze(1)
    
    # Sum over all rules (H axis) to get each feature's net score: (N,)
    global_importance = feature_impact_matrix.sum(dim=0).cpu().numpy()

    # =========================================================================
    # 5. PERCENTAGE NORMALIZATION AND REPORT TABLE
    # =========================================================================
    total = global_importance.sum()
    global_importance_pct = (global_importance / total * 100) if total > 0 else global_importance

    if feature_names is None or len(feature_names) != N:
        feature_names = [f"Feature_{i}" for i in range(N)]

    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Score': global_importance,
        'Importance_Pct': global_importance_pct
    }).sort_values(by='Importance_Score', ascending=False).reset_index(drop=True)

    return df_importance


############################################################################################
############################################################################################
############################################################################################

@torch.no_grad()
def calculate_feature_rule_cooccurrence(model, feature_names=None, threshold=WEIGHT_THRESHOLD):
    """
    Compute two quantities from the topological presence of features in the model's rules,
    without using any weight, importance, or width value:
      1. The total number of rules in which each feature is used.
      2. The number of rules in which each feature pair appears TOGETHER (co-occurrence count).

    Formulation:
        Mask_f = ( |W_f| > threshold )  -> (H, N) binary matrix
        CoOccurrence_Matrix = Mask_f^T @ Mask_f -> (N, N)

    Args:
        model (SCORENet): Trained model.
        feature_names (list of str, optional): Feature names.
        threshold (float): Base absolute weight threshold for a feature to count as 'present'
            in a rule.

    Returns:
        df_cooccurrence (pd.DataFrame): (N, N) co-occurrence frequency matrix.
        df_feature_summary (pd.DataFrame): Per-feature total rule count and unique usage summary.
        df_jaccard (pd.DataFrame): Jaccard similarity matrix (fraction in 0-1) of pair overlap.

    ===================================================================================================
    METHODOLOGICAL NOTES FOR THE PAPER:
    
    1. PURE ARCHITECTURAL TOPOLOGY:
       * Independent of weights, this shows which features the model's architecture always
         groups together.
       * If two features appear together in 28 of 30 rules, the model has built the assumption
         of an "Atomic Feature Pair" directly into its architecture.

    2. EVIDENCE OF PRUNING EFFECT:
       * After iterative pruning (IMP), noisy variables drop to zero rules, while informative
         core variables and their core combinations become directly visible.
    ===================================================================================================
    """
    # 1. Get the first-layer weights of the model (H, N)
    W_f = model.feature_linear.weight.data.abs()
    H, N = W_f.shape

    if feature_names is None or len(feature_names) != N:
        feature_names = [f"Feature_{i}" for i in range(N)]

    # 2. Binary presence matrix (mask): is the feature present in the rule? (1: yes, 0: no)
    binary_mask = (W_f > threshold).float()  # Shape: (H, N)

    # 3. Total number of rules in which each feature appears: (N,)
    feature_total_rule_counts = binary_mask.sum(dim=0).cpu().numpy().astype(int)

    # 4. Co-occurrence matrix: (N, H) @ (H, N) -> (N, N)
    # The matrix product directly gives the number of rules where two features intersect
    cooccurrence_matrix = (binary_mask.T @ binary_mask).cpu().numpy().astype(int)

    # 5. DataFrame conversion (frequency matrix)
    df_cooccurrence = pd.DataFrame(
        cooccurrence_matrix, 
        index=feature_names, 
        columns=feature_names
    )

    # 6. Unique usage summary table
    df_feature_summary = pd.DataFrame({
        'Feature': feature_names,
        'Total_Rule_Count': feature_total_rule_counts,
        'Rule_Coverage_Pct': (feature_total_rule_counts / H) * 100
    }).sort_values(by='Total_Rule_Count', ascending=False).reset_index(drop=True)

    # 7. Jaccard similarity matrix (0-1): intersection / union
    # Formulation: |A intersect B| / (|A| + |B| - |A intersect B|)
    jaccard_matrix = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            intersection = cooccurrence_matrix[i, j]
            union = feature_total_rule_counts[i] + feature_total_rule_counts[j] - intersection
            if union > 0:
                jaccard_matrix[i, j] = intersection / union
            else:
                jaccard_matrix[i, j] = 0.0

    df_jaccard = pd.DataFrame(
        np.round(jaccard_matrix, 2), 
        index=feature_names, 
        columns=feature_names
    )

    return df_cooccurrence, df_feature_summary, df_jaccard

############################################################################################
############################################################################################
############################################################################################


@torch.no_grad()
def analyze_rule_complexity_and_coverage(model, data_loader, threshold=WEIGHT_THRESHOLD):
    """
    Compute the complexity of the active rules (number of variables / rule length), their
    decision strength in the output layer (Output Weight Impact), and their actual
    activation frequency on the dataset (Coverage / Support %).

    For each rule-length level (1-variable, 2-variable, etc.) it also produces a grouped
    summary table with:
      - Total Rule Count,
      - Total Output Strength (Sum of Output Weights),
      - Mean Sample Coverage (%),
      - Composite Impact Score.

    Args:
        model (SCORENet): Trained and compressed deep learning model.
        data_loader (DataLoader): Test/train data loader used to measure rule activations.
        threshold (float): Absolute weight threshold for a feature to count as 'active/present'
            in a rule.

    Returns:
        df_rules (pd.DataFrame): Detailed per-rule metrics table.
        df_length_summary (pd.DataFrame): Summary analysis grouped by rule length.
        stats (dict): Overall summary statistics of the model.

    ===================================================================================================
    METHODOLOGICAL NOTES FOR THE PAPER:
    
    1. COGNITIVE INTERPRETABILITY:
       * In rule-based systems, a rule is understandable if its number of variables stays
         within human short-term memory limits (Miller's Law: 7±2, preferably 1-3).
       * This function quantifies whether the model places its decision power on simple,
         transparent 1-2 variable rules or on complex 15-20 variable rules.

    2. COMBINING THEORETICAL STRENGTH WITH PRACTICAL SUPPORT:
       * Looking only at output weights (|W_o|) can be misleading (a rule may never fire on the data).
       * Looking only at coverage (Support %) is also insufficient (a frequent rule may have little
         effect on the decision).
       * The 'Composite_Impact' metric multiplies both dimensions and shows the true centre of
         gravity of the model.
    ===================================================================================================
    """
    device = model.device
    model.eval()

    # 1. Model parameters
    W_f = model.feature_linear.weight.data.abs()              # (H, N)
    W_o = model.output_linear.weight.data.abs()               # (C, H)
    width = torch.abs(model.base_feature_width.data)   # (H,)
    center = model.base_feature_center.data                   # (H,)
    
    H, N = W_f.shape

    # Total absolute decision strength of each rule in the output layer (L1 norm): (H,)
    rule_output_powers = W_o.sum(dim=0).cpu().numpy()

    # 2. Rule length (number of active features in each rule): (H,)
    rule_lengths = (W_f > threshold).sum(dim=1).cpu().numpy()

    # 3. Scan rule activations over the dataset
    total_samples = 0
    rule_activation_counts = torch.zeros(H, device=device)
    rule_total_activation_sum = torch.zeros(H, device=device)

    for batch in data_loader:
        batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
        batch_x = batch_x.to(device)
        total_samples += batch_x.shape[0]

        # Tent (triangular) activation
        h = model.feature_linear(batch_x)
        act = triangular_activation(h, center, width)  # (B, H)

        # Count samples where the rule is active (produces a score above zero)
        active_mask = (act > 1e-5).float()
        rule_activation_counts += active_mask.sum(dim=0)
        rule_total_activation_sum += act.sum(dim=0)

    # Percentage of the dataset covered by each rule (%): (H,)
    rule_support_pct = (rule_activation_counts.cpu().numpy() / total_samples) * 100.0
    
    # Mean activation intensity of each rule [0, 1]: (H,)
    rule_mean_act_intensity = (rule_total_activation_sum.cpu().numpy() / total_samples)

    # 4. DETAILED PER-RULE DATAFRAME
    df_rules = pd.DataFrame({
        'Rule_ID': [f"Rule_{i+1}" for i in range(H)],
        'Rule_Length': rule_lengths,
        'Output_Power': rule_output_powers,
        'Support_Pct': rule_support_pct,
        'Mean_Intensity': rule_mean_act_intensity,
        'Active_Sample_Count': rule_activation_counts.cpu().numpy().astype(int),
        'Rule_Impact_Score': rule_output_powers * (rule_support_pct / 100.0)
    })

    # 5. SUMMARY TABLE GROUPED BY RULE LENGTH
    # For each rule length (1-variable, 2-variable, ...): total power, mean coverage, and impact
    length_groups = df_rules.groupby('Rule_Length')

    summary_rows = []
    for length_val, group in length_groups:
        count = len(group)
        total_power = group['Output_Power'].sum()
        mean_coverage = group['Support_Pct'].mean()
        composite_impact = total_power * (mean_coverage / 100.0)

        summary_rows.append({
            'Rule_Length': int(length_val),
            'Rule_Count': count,
            'Count_Share_Pct': (count / H) * 100.0,
            'Total_Output_Power': total_power,
            'Power_Share_Pct': 0.0,  # Normalized below
            'Mean_Coverage_Pct': mean_coverage,
            'Composite_Impact': composite_impact,
            'Impact_Share_Pct': 0.0  # Normalized below
        })

    df_length_summary = pd.DataFrame(summary_rows)

    # Normalize percentage shares
    if not df_length_summary.empty:
        total_pwr = df_length_summary['Total_Output_Power'].sum()
        total_imp = df_length_summary['Composite_Impact'].sum()
        
        df_length_summary['Power_Share_Pct'] = (df_length_summary['Total_Output_Power'] / total_pwr * 100.0) if total_pwr > 0 else 0.0
        df_length_summary['Impact_Share_Pct'] = (df_length_summary['Composite_Impact'] / total_imp * 100.0) if total_imp > 0 else 0.0

    # 6. OVERALL STATISTICS SUMMARY DICTIONARY
    outlier_rules = df_rules[df_rules['Support_Pct'] < 0.1]
    
    stats = {
        'total_rules': H,
        'avg_rule_length': float(np.mean(rule_lengths)) if H > 0 else 0.0,
        'median_rule_length': float(np.median(rule_lengths)) if H > 0 else 0.0,
        '1_var_rules': int(np.sum(rule_lengths == 1)),
        '2_var_rules': int(np.sum(rule_lengths == 2)),
        '3plus_var_rules': int(np.sum(rule_lengths >= 3)),
        'outlier_rule_count': len(outlier_rules),
        'avg_coverage_pct': float(np.mean(rule_support_pct)) if H > 0 else 0.0
    }

    return df_rules, df_length_summary, stats

############################################################################################
############################################################################################
############################################################################################

@torch.no_grad()
def evaluate_scorecard_quantization_analysis(
    model, 
    test_loader, 
    scale_factors=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
    weight_threshold=WEIGHT_THRESHOLD,
    max_adaptive_scale=50,
    tolerance_error=0.05
):
    """
    Analyse two quantization regimes for scorecard conversion:
      1. Global (uniform) quantization: single lambda values supplied by the user.
      2. Per-rule adaptive quantization: for each rule, all integer scales in
         [1, max_adaptive_scale] are scanned in one vectorized matrix operation, and the
         SMALLEST (simplest, most readable) local coefficient that satisfies the acceptable
         error threshold (tolerance_error) is selected dynamically.

    Args:
        model (SCORENet): Trained and compressed deep learning model.
        test_loader (DataLoader): Test data loader for measuring accuracy and loss.
        scale_factors (list): Lambda coefficients tried in global mode.
        weight_threshold (float): Base weight threshold treated as zero.
        max_adaptive_scale (int): Maximum multiplier scanned in adaptive mode (ceiling).
        tolerance_error (float): Criterion for choosing the smallest lambda: stop once the
            relative L1 rounding error falls below this threshold, without moving to larger
            numbers. This value is only a fallback default; reported results use the
            dataset-specific budget in ``utils.experiments.DatasetSpec.quantization(...)``.

    Returns:
        df_global (pd.DataFrame): Global scaling sensitivity results.
        df_adaptive_summary (pd.DataFrame): Per-rule adaptive quantization comparison table.
        df_rule_scales (pd.DataFrame): Per-rule automatically selected local lambda coefficients.

    Note:
        Only the input projection ``W_f`` is converted to integers. The output layer
        ``W_o`` stays continuous (paper section "Rule-Level Integer Quantization").
        Thus the scorecard only makes rule conditions integer; class votes are read as
        continuous weights.

    ===================================================================================================
    METHODOLOGICAL NOTES FOR THE PAPER:
    
    1. EXHAUSTIVE VECTORIZED SEARCH:
       * Instead of sampling candidates with linspace, ALL integers in [1, max_adaptive_scale]
         are scanned completely in microseconds with a vectorized matrix product. No optimal
         ratio is skipped.

    2. PARSIMONY & INTERPRETABILITY TRADE-OFF:
       * Searching only for minimum error pushes the algorithm to unnecessarily large numbers
         (e.g. lambda=50).
       * With the 'tolerance_error' mechanism, the model picks the FIRST and SMALLEST lambda
         that reaches sufficient precision (error below 5%) to represent the rule. Scorecard
         scores therefore stay in a minimal range (e.g. [-5, +5]).

    3. EXACT GEOMETRIC PRESERVATION:
       * Multiplying a rule's internal weights by lambda_h does not change the rule's size in
         input space, so the model's decision boundaries remain identical to the float model.
    ===================================================================================================
    """
    device = model.device
    model.eval()

    # Back up the original parameters
    W_f_orig = model.feature_linear.weight.data.clone()
    W_o_orig = model.output_linear.weight.data.clone()
    center_orig = model.base_feature_center.data.clone()
    width_orig = model.base_feature_width.data.clone()

    # -------------------------------------------------------------------------
    # 0. BASELINE: compute the original float model accuracy
    # -------------------------------------------------------------------------
    correct, total = 0, 0
    float_preds, float_logits = [], []
    for batch in test_loader:
        bx = batch[0] if isinstance(batch, (list, tuple)) else batch
        by = batch[1] if isinstance(batch, (list, tuple)) else None
        bx, by = bx.to(device), by.to(device)
        logits = model(bx)
        preds = logits.argmax(dim=1)
        float_preds.append(preds.cpu())
        float_logits.append(logits.detach().cpu())
        correct += (preds == by).sum().item()
        total += by.size(0)
    float_acc = correct / total
    float_preds = torch.cat(float_preds)
    float_logits = torch.cat(float_logits)

    try:
        return _quantization_experiments(
            model=model,
            test_loader=test_loader,
            device=device,
            scale_factors=scale_factors,
            weight_threshold=weight_threshold,
            max_adaptive_scale=max_adaptive_scale,
            tolerance_error=tolerance_error,
            W_f_orig=W_f_orig,
            W_o_orig=W_o_orig,
            center_orig=center_orig,
            width_orig=width_orig,
            float_acc=float_acc,
            float_preds=float_preds,
            float_logits=float_logits,
            total=total,
        )
    finally:
        # The analysed model is never left in a quantized state.
        model.feature_linear.weight.data = W_f_orig
        model.output_linear.weight.data = W_o_orig
        model.base_feature_center.data = center_orig
        model.base_feature_width.data = width_orig


def _quantization_experiments(
    model,
    test_loader,
    device,
    scale_factors,
    weight_threshold,
    max_adaptive_scale,
    tolerance_error,
    W_f_orig,
    W_o_orig,
    center_orig,
    width_orig,
    float_acc,
    float_preds,
    float_logits,
    total,
):
    """Body of ``evaluate_scorecard_quantization_analysis`` (restore is done outside)."""
    H = W_f_orig.shape[0]

    # =========================================================================
    # REGIME 1: GLOBAL (UNIFORM) QUANTIZATION ANALYSIS
    # =========================================================================
    global_results = [{
        'Mode': 'Global',
        'Scale_Factor': 'Float (Original)',
        'Accuracy': float_acc,
        'Accuracy_Drop': 0.0,
        'Min_Int_Weight': 'N/A',
        'Max_Int_Weight': 'N/A',
        'Mean_Abs_Weight': float(W_f_orig.abs().mean().item()),
        'Max_Int_Span': 'N/A'
    }]

    for scale in scale_factors:
        # In global mode too, only W_f is converted to integers; W_o stays continuous.
        W_f_quant = quantize_projection_uniform(W_f_orig, scale)
        model.feature_linear.weight.data = W_f_quant
        model.output_linear.weight.data = W_o_orig

        q_correct = 0
        for batch in test_loader:
            bx = batch[0] if isinstance(batch, (list, tuple)) else batch
            by = batch[1] if isinstance(batch, (list, tuple)) else None
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            q_correct += (logits.argmax(dim=1) == by).sum().item()

        q_acc = q_correct / total
        int_w_f = integer_weights(W_f_orig.cpu().numpy(), np.full(H, scale))
        min_w, max_w = int(np.min(int_w_f)), int(np.max(int_w_f))

        global_results.append({
            'Mode': 'Global',
            'Scale_Factor': str(scale),
            'Accuracy': q_acc,
            'Accuracy_Drop': float_acc - q_acc,
            'Min_Int_Weight': min_w,
            'Max_Int_Weight': max_w,
            'Mean_Abs_Weight': float(np.mean(np.abs(int_w_f))),
            'Max_Int_Span': max_w - min_w
        })

    df_global = pd.DataFrame(global_results)

    # Restore the model to its original values
    model.feature_linear.weight.data = W_f_orig.clone()
    model.output_linear.weight.data = W_o_orig.clone()

    # =========================================================================
    # REGIME 2: PER-RULE ADAPTIVE QUANTIZATION (METHOD 3: EXHAUSTIVE VECTORIZED SEARCH)
    # =========================================================================
    # The scale search lives in a single helper; paper figures call the same function,
    # so the numbers in the figures cannot drift from the code.
    adaptive_scales, rule_quant_errors = per_rule_scales(
        W_f_orig.cpu().numpy(),
        max_scale=max_adaptive_scale,
        tolerance=tolerance_error,
        weight_threshold=weight_threshold,
    )

    # Apply adaptive quantization to the model.
    # Only W_f is rounded to integers. The output layer W_o stays continuous: the paper
    # defines class votes as "transparent continuous voting weights" (W_o is not quantized).
    W_f_adaptive = quantize_projection(W_f_orig, adaptive_scales)
    model.feature_linear.weight.data = W_f_adaptive
    model.output_linear.weight.data = W_o_orig

    adapt_correct = 0
    adapt_preds, adapt_logits = [], []
    for batch in test_loader:
        bx = batch[0] if isinstance(batch, (list, tuple)) else batch
        by = batch[1] if isinstance(batch, (list, tuple)) else None
        bx, by = bx.to(device), by.to(device)
        logits = model(bx)
        preds = logits.argmax(dim=1)
        adapt_preds.append(preds.cpu())
        adapt_logits.append(logits.detach().cpu())
        adapt_correct += (preds == by).sum().item()

    adapt_acc = adapt_correct / total
    adapt_preds = torch.cat(adapt_preds)
    max_logit_drift = float((torch.cat(adapt_logits) - float_logits).abs().max().item())
    fidelity = float((adapt_preds == float_preds).float().mean().item()) * 100.0
    int_w_adapt = integer_weights(W_f_orig.cpu().numpy(), adaptive_scales)

    df_adaptive_summary = pd.DataFrame([
        {
            'Method': 'Original Float Model',
            'Accuracy (%)': float_acc * 100.0,
            'Accuracy Loss (%)': 0.0,
            'Mean Scale (λ)': 1.0,
            'Min Integer Score': 'N/A',
            'Max Integer Score': 'N/A',
            'Mean Weight Magnitude': float(W_f_orig.abs().mean().item()),
            'Fidelity (%)': 100.0,
            'Max Logit Drift': 0.0,
        },
        {
            'Method': 'Rule-Based Adaptive Quantization',
            'Accuracy (%)': adapt_acc * 100.0,
            'Accuracy Loss (%)': (float_acc - adapt_acc) * 100.0,
            'Mean Scale (λ)': float(np.mean(adaptive_scales)),
            'Min Integer Score': int(np.min(int_w_adapt)),
            'Max Integer Score': int(np.max(int_w_adapt)),
            'Mean Weight Magnitude': float(np.mean(np.abs(int_w_adapt))),
            'Fidelity (%)': fidelity,
            'Max Logit Drift': max_logit_drift,
        }
    ])

    df_rule_scales = pd.DataFrame({
        'Rule_ID': [f"Rule_{i+1}" for i in range(H)],
        'Optimal_Scale_Factor': adaptive_scales.astype(int),
        'Rel_Quant_Error': np.round(rule_quant_errors, 4),
        'Min_Int': np.min(int_w_adapt, axis=1).astype(int),
        'Max_Int': np.max(int_w_adapt, axis=1).astype(int)
    })

    return df_global, df_adaptive_summary, df_rule_scales


############################################################################################
############################################################################################
############################################################################################
