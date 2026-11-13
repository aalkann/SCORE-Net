import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.analysis import *


####################################################################################
####################################################################################
####################################################################################

@torch.no_grad()
def plot_exact_feature_impact(
    model,
    single_record,
    feature_idx,
    f_min,
    f_max,
    feature_name=None,
    class_names=None,
    selected_class_indexes=None
):
    """
    Compute and plot the effect of one feature on the model's decision as an
    exact partial dependence curve, without brute-force simulation.

    Args:
        model (SCORENet): Trained piecewise-linear model.
        single_record (Tensor/Array/List): Reference record (1-D or 1xN).
        feature_idx (int): Column index of the feature to analyse.
        f_min (float): Minimum feature value shown on the x-axis.
        f_max (float): Maximum feature value shown on the x-axis.
        feature_name (str, optional): Feature name used in the plot.
        class_names (list, optional): Class labels shown in the legend.
        selected_class_indexes (list, optional): Classes to plot.

    Notes:
    * White-box proof: the curves are not estimated by running the model many
      times. The breakpoints of the triangular activation are found
      algebraically.
    * O(1) cost and zero fidelity loss: matrix operations give the exact curve
      and miss no threshold, unlike simulation-based PDP or SHAP.
    * Practical use: the plot answers questions such as "at which income does
      the risk score jump?" with an exact threshold.
    """
    device = model.device

    # =========================================================================
    # 1. INPUT TYPE CHECK AND TENSOR CONVERSION
    # =========================================================================
    if not isinstance(single_record, torch.Tensor):
        single_record = torch.tensor(single_record, dtype=torch.float32)

    # Expand a vector (N,) to the matrix shape (1, N) expected by PyTorch
    if single_record.dim() == 1:
        single_record = single_record.unsqueeze(0)

    single_record = single_record.to(device)

    # Fill in default arguments (names, indices)
    if feature_name is None:
        feature_name = f"Feature {feature_idx}"

    C_classes = model.output_linear.weight.shape[0]

    if selected_class_indexes is None:
        selected_class_indexes = list(range(C_classes))

    if class_names is None:
        class_names = [f"Class {c}" for c in range(C_classes)]

    # =========================================================================
    # 2. READ THE MODEL'S TRAINED PARAMETERS
    # =========================================================================
    W_f = model.feature_linear.weight.data         # Feature weights (H, N)
    W_o = model.output_linear.weight.data          # Class weights (C, H)
    C = model.base_feature_center.data             # Rule centres (H)
    W = torch.abs(model.base_feature_width.data)   # Rule widths (H)

    # Take the output bias into account if the model has one
    has_bias = getattr(model.output_linear, 'bias', None) is not None
    bias_o = model.output_linear.bias.data if has_bias else 0

    # =========================================================================
    # 3. BUILD THE LINEAR EQUATION (y = mx + b) FOR THE FEATURE
    # =========================================================================
    m = W_f[:, feature_idx] # Rule weights of the feature (slope)
    
    # Current linear combination of the rules
    h_current = model.feature_linear(single_record)[0]
    
    # Contribution of all other variables (intercept)
    b = h_current - (m * single_record[0, feature_idx])

    # =========================================================================
    # 4. ANALYTIC BREAKPOINT COMPUTATION
    # =========================================================================
    # Keep only active rules that use this feature (m != 0)
    valid_rules = (m != 0)
    m_v, b_v, C_v, W_v = m[valid_rules], b[valid_rules], C[valid_rules], W[valid_rules]

    # Three breakpoints of the triangular activation
    v_left = (C_v - W_v - b_v) / m_v   # Left edge (activation starts)
    v_center = (C_v - b_v) / m_v       # Centre (activation peaks)
    v_right = (C_v + W_v - b_v) / m_v  # Right edge (activation ends)

    # Merge the points and keep those inside the given min-max range
    breakpoints = torch.cat([v_left, v_center, v_right])
    breakpoints = breakpoints[(breakpoints >= f_min) & (breakpoints <= f_max)]

    # Include the current value and the range bounds
    orig_val = single_record[0, feature_idx].item()
    all_points = torch.cat([breakpoints, torch.tensor([f_min, f_max, orig_val], device=device)])

    # Remove duplicates and sort ascending
    sorted_v = torch.sort(torch.unique(torch.round(all_points * 1e4) / 1e4)).values

    # =========================================================================
    # 5. EXACT SCORES BY VECTORISED MATRIX MULTIPLICATION (ZERO SIMULATION)
    # =========================================================================
    # Broadcast to (number of points x number of rules)
    V_matrix = sorted_v.unsqueeze(1)
    m_matrix = m.unsqueeze(0)
    b_matrix = b.unsqueeze(0)

    # Linear h values at all points
    H_matrix = (V_matrix * m_matrix) + b_matrix

    # Exact activation strengths at all points (matrix form of the ReLU formula)
    C_matrix = C.unsqueeze(0)
    W_matrix = W.unsqueeze(0)
    Act_matrix = torch.relu(W_matrix - torch.abs(C_matrix - H_matrix)) / (W_matrix + 1e-8)

    # Multiply activations by class weights to obtain the final logits
    Logits_matrix = torch.matmul(Act_matrix, W_o.T) + bias_o

    # =========================================================================
    # 6. PLOTTING AND LEGEND SETTINGS
    # =========================================================================
    X_plot = sorted_v.cpu().numpy()
    Y_plot = Logits_matrix.cpu().numpy()

    # Widen the figure to fit the legend on the right
    plt.figure(figsize=(11, 6)) 

    # Colour palette suitable for academic publications
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']

    # Plot only the classes selected by the user
    for idx, c in enumerate(selected_class_indexes):
        plt.plot(X_plot, Y_plot[:, c], label=class_names[idx],
                 linewidth=2.5, color=colors[idx % len(colors)])

    # Dashed vertical line marking the current value
    plt.axvline(x=orig_val, color='black', linestyle='--', linewidth=2, label=f'Current\n({orig_val:.2f})')

    plt.title(f"'{feature_name}' Change: Exact Effect on Class Scores", fontsize=14, fontweight='bold')
    plt.xlabel(f"{feature_name} Value", fontsize=12)
    plt.ylabel("Model Decision Score", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)

    # Anchor the legend outside the axes on the right so it does not cover the plot
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, shadow=True)

    # Tighten the layout so the plot and legend are not clipped
    plt.tight_layout()
    plt.show()


####################################################################################
####################################################################################
####################################################################################

@torch.no_grad()
def plot_simulated_feature_impact(
    model,
    single_record,
    feature_idx,
    f_min,
    f_max,
    steps=1000,
    feature_name=None,
    class_names=None,
    selected_class_indexes=None
):
    """
    Compute and plot the effect of one feature by brute-force simulation, i.e.
    by running the model many times from scratch.

    Args:
        model (SCORENet): Trained model.
        single_record (Tensor/Array/List): Reference record.
        feature_idx (int): Column index of the feature to analyse.
        f_min (float): Minimum simulation value on the x-axis.
        f_max (float): Maximum simulation value on the x-axis.
        steps (int): Number of simulation points (resolution). Default: 1000.
        feature_name (str, optional): Feature name used in the plot.
        class_names (list, optional): Class labels shown in the legend.
        selected_class_indexes (list, optional): Classes to plot.

    Notes:
    * Sanity check: this simulation serves as the ground truth for the exact
      analytic method.
    * Fidelity proof: overlaying the simulated and analytic curves shows that
      they coincide, which supports the claim that the analytic method makes
      no approximation.
    * Cost comparison: this function performs ``steps`` forward passes, while
      the analytic method returns the same result in O(1) without running the
      model. The runtimes can be compared in the paper.
    """
    device = model.device
    model.eval()

    # =========================================================================
    # 1. INPUT TYPE CHECK AND TENSOR CONVERSION
    # =========================================================================
    if not isinstance(single_record, torch.Tensor):
        single_record = torch.tensor(single_record, dtype=torch.float32)

    if single_record.dim() == 1:
        single_record = single_record.unsqueeze(0)

    single_record = single_record.to(device)

    # Default arguments
    if feature_name is None:
        feature_name = f"Feature {feature_idx}"

    C_classes = model.output_linear.weight.shape[0]

    if selected_class_indexes is None:
        selected_class_indexes = list(range(C_classes))

    if class_names is None:
        class_names = [f"Class {c}" for c in range(C_classes)]

    # =========================================================================
    # 2. BUILD THE SIMULATION GRID
    # =========================================================================
    # Split [f_min, f_max] into 'steps' equally spaced points
    test_values = np.linspace(f_min, f_max, steps)

    # Matrix that will hold the logits (class scores) at each step
    simulated_logits = np.zeros((steps, C_classes))

    # Original value of the feature (shown as a vertical line)
    orig_val = single_record[0, feature_idx].item()

    # =========================================================================
    # 3. BRUTE-FORCE SIMULATION LOOP
    # =========================================================================
    # Run the model end to end 'steps' times (computationally expensive)
    for i, val in enumerate(test_values):
        # Copy the original record to keep it unchanged
        temp_record = single_record.clone()

        # Replace only the analysed feature with the simulation point
        temp_record[0, feature_idx] = val

        # Run a forward pass
        logits = model(temp_record)

        # Store the class scores
        simulated_logits[i, :] = logits[0].cpu().numpy()

    # =========================================================================
    # 4. PLOTTING AND LEGEND SETTINGS (same as the analytic version)
    # =========================================================================
    plt.figure(figsize=(11, 6))

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']

    # Plot only the selected classes
    for idx, c in enumerate(selected_class_indexes):
        # Add the simulation suffix to distinguish it from the analytic curve.
        plt.plot(test_values, simulated_logits[:, c], label=f"{class_names[idx]} (Simulation)",
                 linewidth=2.5, color=colors[idx % len(colors)], linestyle='-')

    # Dashed vertical line marking the current value
    plt.axvline(x=orig_val, color='black', linestyle='--', linewidth=2, label=f'Current\n({orig_val:.2f})')

    # The title marks this as the simulation method
    plt.title(f"'{feature_name}' Change: Effect on Class Scores (SIMULATION)", fontsize=14, fontweight='bold')
    plt.xlabel(f"{feature_name} Value", fontsize=12)
    plt.ylabel("Model Decision Score", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)

    # Anchor the legend outside the axes on the right so it does not cover the plot
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, shadow=True)

    plt.tight_layout()
    plt.show()


####################################################################################
####################################################################################
####################################################################################

def plot_global_feature_importance(
    model, 
    data_loader=None, 
    feature_names=None, 
    top_feature=0, 
    show_cumulative=True
):
    """
    Plot the analytic global feature importance scores of a SCORENet model as
    a publication-quality horizontal bar chart, with an optional Pareto curve.

    Args:
        model (SCORENet): Trained model.
        data_loader (DataLoader, optional): Loader used to measure rule
                                            coverage.
        feature_names (list, optional): Feature names.
        top_feature (int, optional): Limit to the N most important features.
                                     (0 plots all features.)
        show_cumulative (bool, optional): Show the cumulative importance
                                          (Pareto curve) on a secondary axis.

    Notes:
    * The scores come directly from the model's own weights (W_f, W_o) and data
      coverage, so no approximate tool such as SHAP or permutation importance
      is needed.
    * After pruning, noise features drop to 0% and only informative variables
      remain at the top.
    """
    # 1. Compute analytic importance scores
    df_importance = calculate_global_feature_importance(
        model, 
        data_loader=data_loader, 
        feature_names=feature_names
    )
    
    total_feature_count = len(df_importance)
    
    # 2. Top-N filtering
    if top_feature > 0 and top_feature < total_feature_count:
        plot_df = df_importance.head(top_feature).copy()
        title_suffix = f" (First {top_feature} / {total_feature_count} Features)"
    else:
        plot_df = df_importance.copy()
        title_suffix = f" (Total {total_feature_count} Features)"

    # Cumulative percentage (Pareto analysis)
    plot_df['Cumulative_Pct'] = plot_df['Importance_Pct'].cumsum()

    # Plot order: reverse so the highest importance appears at the top
    plot_df_sorted = plot_df.sort_values(by='Importance_Score', ascending=True).reset_index(drop=True)

    # 3. Set up the figure (dynamic height)
    fig_height = max(6, len(plot_df_sorted) * 0.35)
    fig, ax1 = plt.subplots(figsize=(12, fig_height))
    sns.set_style("whitegrid")

    # Colour palette (Viridis gradient by importance)
    colors = sns.color_palette("viridis", n_colors=len(plot_df_sorted))

    # 4. Draw the horizontal bar chart
    bars = ax1.barh(
        plot_df_sorted['Feature'], 
        plot_df_sorted['Importance_Pct'], 
        color=colors, 
        edgecolor='black', 
        alpha=0.88,
        height=0.65
    )

    # 5. Add value labels to the bars
    max_pct = plot_df_sorted['Importance_Pct'].max()
    for bar, (_, row) in zip(bars, plot_df_sorted.iterrows()):
        width = bar.get_width()
        # Label format: "%14.2 (Score: 1.85)"
        label_text = f" %{row['Importance_Pct']:.2f}  [Skor: {row['Importance_Score']:.2f}]"
        
        ax1.text(
            width + (max_pct * 0.01), 
            bar.get_y() + bar.get_height() / 2, 
            label_text,
            va='center', 
            ha='left', 
            fontsize=9.5, 
            fontweight='bold',
            color='#2c3e50'
        )

    # 6. Axis and title adjustments
    ax1.set_title(f"Analytical Global Feature Importance Distribution{title_suffix}", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Percentage Decision Contribution (%)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Features", fontsize=11, fontweight='bold')
    ax1.set_xlim(0, max_pct * 1.25) # Leave room on the right for the labels
    ax1.grid(axis='x', linestyle=':', alpha=0.6)

    # Remove top and right spines (clean minimalist look)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # 7. Cumulative line (Pareto line, optional)
    if show_cumulative and len(plot_df_sorted) > 2:
        ax2 = ax1.twiny()
        ax2.plot(
            plot_df_sorted['Cumulative_Pct'], 
            plot_df_sorted['Feature'], 
            color='#e74c3c', 
            marker='o', 
            linewidth=2, 
            linestyle='--', 
            alpha=0.75, 
            label='Cumulative Share (%)'
        )
        ax2.set_xlabel("Cumulative Total Share (%)", color='#e74c3c', fontsize=10, fontweight='bold')
        ax2.tick_params(axis='x', labelcolor='#e74c3c')
        ax2.set_xlim(0, 105)
        ax2.grid(False) # Disable the second grid
        ax2.spines['top'].set_color('#e74c3c')
        ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

####################################################################################
####################################################################################
####################################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_feature_rule_cooccurrence_dashboard(
    model, 
    feature_names=None, 
    threshold=1e-4, 
    top_n_features=15
):
    """
    Visualise the structural co-occurrence of features across the model's
    rules in three independent figures: individual coverage, a frequency
    matrix, and the Jaccard association index.

    Args:
        model (SCORENet): Trained piecewise-linear model.
        feature_names (list of str, optional): Feature names.
        threshold (float): Minimum absolute weight for a feature to count as
                           active in a rule.
        top_n_features (int): Keep only the N features that appear in the most
                              rules, to keep the matrices readable (0 keeps
                              all).

    Notes:
    * Figure 1 (rule coverage): number of rules in which each feature survives.
      After iterative pruning, noise variables drop to length 0.
    * Figure 2 (co-occurrence frequency): integer count of rules in which each
      feature pair appears together.
    * Figure 3 (Jaccard index): normalised dependence,
      |A intersect B| / |A union B| * 100. Values near 100% mean the model
      never separates the two features.
    """
    # =========================================================================
    # 1. RUN THE TOPOLOGY COMPUTATION
    # =========================================================================
    # Retrieve the frequency matrix, the per-feature summary and the Jaccard matrix
    df_coocc, df_summary, df_jaccard = calculate_feature_rule_cooccurrence(
        model, 
        feature_names=feature_names, 
        threshold=threshold
    )

    total_rules = model.H
    
    # =========================================================================
    # 2. TOP-N FILTERING FOR A CLEANER LAYOUT
    # =========================================================================
    # If there are many features, keep only the N most active ones
    if 0 < top_n_features < len(df_summary):
        selected_features = df_summary.head(top_n_features)['Feature'].tolist()
        df_summary_plot = df_summary.head(top_n_features).copy()
        df_coocc_plot = df_coocc.loc[selected_features, selected_features].copy()
        df_jaccard_plot = df_jaccard.loc[selected_features, selected_features].copy()
        title_suffix = f" (En Aktif {top_n_features} Features)"
    else:
        selected_features = df_summary['Feature'].tolist()
        df_summary_plot = df_summary.copy()
        df_coocc_plot = df_coocc.copy()
        df_jaccard_plot = df_jaccard.copy()
        title_suffix = " (All Features)"

    # Reverse the summary table so the feature with most rules is at the top
    df_summary_plot_sorted = df_summary_plot.sort_values(by='Total_Rule_Count', ascending=True)
    n_feats = len(selected_features)

    # -------------------------------------------------------------------------
    # FIGURE 1: RULE COUNT AND COVERAGE PER FEATURE (separate figure)
    # -------------------------------------------------------------------------
    height_bar = max(5.0, n_feats * 0.3)
    plt.figure(figsize=(12, height_bar))
    sns.set_style("whitegrid")

    bars = plt.barh(
        df_summary_plot_sorted['Feature'], 
        df_summary_plot_sorted['Total_Rule_Count'], 
        color=sns.color_palette("Blues_r", n_colors=len(df_summary_plot_sorted)),
        edgecolor='black',
        alpha=0.85
    )
    
    # Add value labels to the right of the bars
    max_count = df_summary_plot_sorted['Total_Rule_Count'].max()
    for bar, (_, row) in zip(bars, df_summary_plot_sorted.iterrows()):
        width = bar.get_width()
        label_text = f" {int(row['Total_Rule_Count'])} Rule (%{row['Rule_Coverage_Pct']:.1f})"
        plt.text(
            width + (max_count * 0.01), 
            bar.get_y() + bar.get_height() / 3, 
            label_text, 
            va='center', 
            ha='left', 
            fontsize=10, 
            fontweight='bold', 
            color='#1a252f'
        )

    plt.title(f"A) Individual Rule Presence (Total Rule Count and Coverage Share %){title_suffix}\n(Total Rule Capacity: H={total_rules} | Threshold: {threshold})", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Number of Rules Containing It", fontsize=11, fontweight='bold')
    plt.ylabel("Feature Name", fontsize=11, fontweight='bold')
    plt.xlim(0, max(max_count * 1.25, total_rules * 0.5))
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # FIGURE 2: CO-OCCURRENCE FREQUENCY MATRIX (separate figure)
    # -------------------------------------------------------------------------
    height_heatmap = max(6.5, n_feats * 0.30)
    plt.figure(figsize=(11, height_heatmap))
    sns.set_style("white")

    # Use an independent copy to avoid a NumPy read-only view error.
    # Exclude the diagonal from vmax so that a feature's own count does not
    # flatten the contrast between pairs.
    coocc_array = df_coocc_plot.to_numpy(copy=True)
    np.fill_diagonal(coocc_array, 0)
    vmax_coocc = max(1, int(coocc_array.max()))

    sns.heatmap(
        df_coocc_plot, 
        annot=True, 
        fmt="d", 
        cmap="YlGnBu", 
        cbar_kws={'label': 'Co-occurrence Count (Number of Rules)'},
        linewidths=0.3, 
        linecolor='white',
        vmax=vmax_coocc * 1.2,
        annot_kws={"size": 9}
    )
    plt.title(f"B) Co-occurrence Frequency Matrix (In how many rules do they appear together?){title_suffix}\n(Diagonal: individual total rule count)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Features", fontsize=11, fontweight='bold')
    plt.ylabel("Features", fontsize=11, fontweight='bold')
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # FIGURE 3: JACCARD ASSOCIATION MATRIX (separate figure)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(11, height_heatmap))
    sns.set_style("white")

    # Colour scale fixed to 0-100% for the full association analysis
    sns.heatmap(
        df_jaccard_plot, 
        annot=True, 
        fmt=".0f", 
        cmap="mako", 
        cbar_kws={'label': 'Inseparability / Agreement Index (%)'},
        linewidths=0.3, 
        linecolor='white',
        vmin=0, 
        vmax=100,
        annot_kws={"size": 9}
    )
    plt.title(f"C) Jaccard Co-occurrence Index (%){title_suffix}\n(How inseparable is the pair? | Formula: |A ∩ B| / |A ∪ B| * 100)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Features", fontsize=11, fontweight='bold')
    plt.ylabel("Features", fontsize=11, fontweight='bold')
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



####################################################################################
####################################################################################
####################################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_rule_complexity_and_impact_dashboard(
    model, 
    data_loader, 
    threshold=1e-4
):
    """
    Use the output of analyze_rule_complexity_and_coverage to produce three
    independent figures that relate rule complexity (number of variables) to
    output power and data coverage.

    Args:
        model (SCORENet): Trained model.
        data_loader (DataLoader): Dataset loader.
        threshold (float): Threshold for active features inside a rule.

    Notes:
    * Figure 1 (rule count and output power share): number of rules per rule
      length, next to the summed absolute output weight of those rules. Shows
      whether the pruned model prefers short rules.
    * Figure 2 (mean data coverage %): how much of the data each rule-length
      group covers on average (generality vs. specialisation). Rules with 1-2
      variables usually cover more.
    * Figure 3 (combined effective impact %): output weight times mean
      coverage, i.e. the effective decision load per rule-length group.
    """
    # 1. Run the analysis function
    df_rules, df_summary, stats = analyze_rule_complexity_and_coverage(
        model, 
        data_loader, 
        threshold=threshold
    )

    if df_summary.empty:
        print("Warning: no rule data to plot.")
        return

    # Print the summary metrics to the console
    print("\n" + "=" * 80)
    print("RULE COMPLEXITY AND EFFECTIVE IMPACT SUMMARY")
    print("=" * 80)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"➜ {k:<28s}: {v:.2f}")
        else:
            print(f"➜ {k:<28s}: {v}")
    print("=" * 80 + "\n")

    # Plot settings (width grows with the number of groups, fixed generous height)
    x_ticks = df_summary['Rule_Length'].astype(str).tolist()
    n_groups = len(df_summary)
    fig_width = max(11, n_groups * 1.5)
    fig_height = 14.0  # Generous vertical space

    # -------------------------------------------------------------------------
    # FIGURE 1: RULE COUNT AND OUTPUT POWER (separate figure)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(fig_width, fig_height))
    sns.set_style("whitegrid")
    
    colors_count = sns.color_palette("Blues_r", n_colors=n_groups)
    bars1 = plt.bar(
        x_ticks, 
        df_summary['Rule_Count'], 
        color=colors_count, 
        edgecolor='black', 
        alpha=0.88, 
        width=0.55
    )

    # Place labels above the bars
    max_count = df_summary['Rule_Count'].max()
    for bar, (_, row) in zip(bars1, df_summary.iterrows()):
        h = bar.get_height()
        label_text = f"{int(row['Rule_Count'])} rules\n(Power: %{row['Power_Share_Pct']:.1f})"
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            h + (max_count * 0.025), 
            label_text, 
            ha='center', 
            va='bottom', 
            fontsize=18, 
            fontweight='bold', 
            color='#1a252f'
        )

    plt.title(
        f"A) Rule Complexity Distribution and Output Decision Power (%)\n"
        f"(Total rules: H={stats['total_rules']} | Mean length: {stats['avg_rule_length']:.2f} variables)",
        fontsize=16, 
        fontweight='bold', 
        pad=22
    )
    plt.xlabel("Active Variables per Rule (Rule Length)", fontsize=14, fontweight='bold', labelpad=12)
    plt.ylabel("Rule Adedi", fontsize=14, fontweight='bold', labelpad=12)
    plt.xticks(fontsize=13, fontweight='bold')
    plt.yticks(fontsize=13)
    plt.ylim(0, max_count * 1.30)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # FIGURE 2: MEAN DATA COVERAGE (separate figure)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(fig_width, fig_height))
    sns.set_style("whitegrid")

    colors_cov = sns.color_palette("Greens_r", n_colors=n_groups)
    bars2 = plt.bar(
        x_ticks, 
        df_summary['Mean_Coverage_Pct'], 
        color=colors_cov, 
        edgecolor='black', 
        alpha=0.88, 
        width=0.55
    )

    max_cov = df_summary['Mean_Coverage_Pct'].max()
    for bar, (_, row) in zip(bars2, df_summary.iterrows()):
        h = bar.get_height()
        label_text = f"%{row['Mean_Coverage_Pct']:.1f}\nKapsama"
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            h + (max(max_cov, 1.0) * 0.025), 
            label_text, 
            ha='center', 
            va='bottom', 
            fontsize=18, 
            fontweight='bold', 
            color='#145a32'
        )

    plt.title(
        "B) Mean Data Coverage by Rule-Length Group (% Support)\n"
        "(How often do rules fire on average in the dataset?)",
        fontsize=16, 
        fontweight='bold', 
        pad=22
    )
    plt.xlabel("Active Variables per Rule (Rule Length)", fontsize=14, fontweight='bold', labelpad=12)
    plt.ylabel("Mean Data Coverage (%)", fontsize=14, fontweight='bold', labelpad=12)
    plt.xticks(fontsize=13, fontweight='bold')
    plt.yticks(fontsize=13)
    plt.ylim(0, max(max_cov * 1.30, 10))
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # FIGURE 3: COMBINED EFFECTIVE DECISION IMPACT (separate figure)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(fig_width, fig_height))
    sns.set_style("whitegrid")

    colors_imp = sns.color_palette("mako", n_colors=n_groups)
    bars3 = plt.bar(
        x_ticks, 
        df_summary['Impact_Share_Pct'], 
        color=colors_imp, 
        edgecolor='black', 
        alpha=0.88, 
        width=0.55
    )

    max_imp = df_summary['Impact_Share_Pct'].max()
    for bar, (_, row) in zip(bars3, df_summary.iterrows()):
        h = bar.get_height()
        label_text = f"%{row['Impact_Share_Pct']:.1f}\nPay"
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            h + (max(max_imp, 1.0) * 0.025), 
            label_text, 
            ha='center', 
            va='bottom', 
            fontsize=18, 
            fontweight='bold', 
            color='#1a252f'
        )

    plt.title(
        "C) Composite Effective Decision Impact Share (% Impact Share)\n"
        "(Formula: Total Output Weight x Mean Coverage)",
        fontsize=16, 
        fontweight='bold', 
        pad=22
    )
    plt.xlabel("Active Variables per Rule (Rule Length)", fontsize=14, fontweight='bold', labelpad=12)
    plt.ylabel("Net Contribution to Model Decisions (%)", fontsize=14, fontweight='bold', labelpad=12)
    plt.xticks(fontsize=13, fontweight='bold')
    plt.yticks(fontsize=13)
    plt.ylim(0, max(max_imp * 1.30, 10))
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

####################################################################################
####################################################################################
####################################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_scorecard_quantization_dashboard(
    model, 
    test_loader, 
    scale_factors=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
    weight_threshold=1e-4,
    max_adaptive_scale=50,
    tolerance_error=0.05
):
    """
    Run the scorecard quantization analysis and visualise the performance,
    fidelity and integer score ranges of the global uniform and rule-based
    adaptive methods in a console report and three independent figures.

    Args:
        model (SCORENet): Trained model.
        test_loader (DataLoader): Test set loader.
        scale_factors (list): Scale factors tested in global mode.
        weight_threshold (float): Base weight threshold.
        max_adaptive_scale (int): Upper scale bound in adaptive mode.
        tolerance_error (float): Accepted L1 rounding error.

    Notes:
    * Figure 1 (global sensitivity curve): rounding error decreases as the
      single lambda grows, and accuracy approaches the float reference. The
      reference is measured from the model at call time, not fixed here.
    * Figure 2 (method comparison): rule-based adaptive quantization keeps
      fidelity within a narrow integer band that a human can add up, unlike
      the very wide ranges produced by global methods.
    * Figure 3 (per-rule optimal lambda): most rules can be quantized with a
      small coefficient; only a few sensitive rules need larger factors. Wide
      ceilings (max_adaptive_scale >= 100) are binned for readability.
    """
    # 1. Run the analysis function
    df_global, df_adapt_summary, df_rule_scales = evaluate_scorecard_quantization_analysis(
        model=model, 
        test_loader=test_loader, 
        scale_factors=scale_factors,
        weight_threshold=weight_threshold,
        max_adaptive_scale=max_adaptive_scale,
        tolerance_error=tolerance_error
    )

    # -------------------------------------------------------------------------
    # CONSOLE REPORT
    # -------------------------------------------------------------------------
    float_acc = df_global[df_global['Scale_Factor'] == 'Float (Original)']['Accuracy'].values[0] * 100.0
    adapt_row = df_adapt_summary[df_adapt_summary['Method'] == 'Rule-Based Adaptive Quantization'].iloc[0]
    adapt_acc = adapt_row['Accuracy (%)']
    adapt_drop = adapt_row['Accuracy Loss (%)']

    # TWO DIFFERENT METRICS - do not confuse them:
    #   * accuracy_retention  = Acc(quant) / Acc(float)  -> "accuracy retention"
    #   * prediction_fidelity = P(y_hat_quant == y_hat_float)  -> the paper's
    #     "fidelity" (Eq. fidelity). They are not the same: accuracy depends on
    #     class balance, fidelity is per-sample prediction agreement.
    accuracy_retention = (adapt_acc / (float_acc + 1e-8)) * 100.0
    prediction_fidelity = (
        float(adapt_row['Fidelity (%)']) if 'Fidelity (%)' in adapt_row.index else float('nan')
    )
    max_logit_drift = (
        float(adapt_row['Max Logit Drift']) if 'Max Logit Drift' in adapt_row.index else float('nan')
    )

    # Prepare the global analysis table
    df_global_detail = df_global.copy()
    df_global_detail['Accuracy (%)'] = df_global_detail['Accuracy'] * 100.0
    df_global_detail['Loss (%)'] = df_global_detail['Accuracy_Drop'] * 100.0
    df_global_detail['Accuracy Retention (%)'] = (df_global_detail['Accuracy'] / (df_global_detail['Accuracy'].iloc[0] + 1e-8)) * 100.0
    
    # Format the score range
    df_global_detail['Score Range [Min, Max]'] = df_global_detail.apply(
        lambda r: f"[{r['Min_Int_Weight']}, {r['Max_Int_Weight']}]" if r['Min_Int_Weight'] != 'N/A' else 'Original Float',
        axis=1
    )

    df_global_print = df_global_detail[[
        'Scale_Factor', 'Accuracy (%)', 'Loss (%)', 'Accuracy Retention (%)',
        'Score Range [Min, Max]', 'Max_Int_Span', 'Mean_Abs_Weight'
    ]].rename(columns={
        'Scale_Factor': 'Scale (λ)',
        'Max_Int_Span': 'Score Width',
        'Mean_Abs_Weight': 'Mean Coefficient Size'
    })

    print("\n" + "=" * 105)
    print("                     SCORECARD QUANTIZATION AND MODEL FIDELITY REPORT")
    print("=" * 105)
    print(f"-> Original Float Model Accuracy (Reference)     : %{float_acc:.3f}")
    print(f"-> Adaptive Scorecard Accuracy                 : %{adapt_acc:.3f}")
    print(f"-> Accuracy Drop                              : %{adapt_drop:.3f}")
    print(f"-> Accuracy Retention (Acc_quant / Acc_float) : %{accuracy_retention:.2f}  <- accuracy ratio, not fidelity")
    print(f"➜ Prediction Agreement (Fidelity, Eq. fidelity)          : %{prediction_fidelity:.2f}  <- the fidelity reported in the paper")
    print(f"-> Max Logit Drift                          : {max_logit_drift:.4e}")
    print(f"-> Integer Score Band (Scorecard Range)      : [{adapt_row['Min Integer Score']}, {adapt_row['Max Integer Score']}] (width: {int(adapt_row['Max Integer Score']) - int(adapt_row['Min Integer Score'])})")
    print(f"-> Mean / Median Scale Factor per Rule       : λ_mean={float(adapt_row['Mean Scale (λ)']):.2f} | λ_median={df_rule_scales['Optimal_Scale_Factor'].median():.1f}")
    print("-" * 105)
    print("1) DETAILED PERFORMANCE OF GLOBAL SCALE FACTORS (λ):")
    print(df_global_print.to_string(index=False))
    print("-" * 105)
    print("2) METHOD COMPARISON SUMMARY:")
    print(df_adapt_summary.to_string(index=False))
    print("=" * 105 + "\n")

    # -------------------------------------------------------------------------
    # FIGURE 1: GLOBAL SCALE FACTOR (lambda) SENSITIVITY CURVE (separate figure)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 7.5))
    sns.set_style("whitegrid")

    df_plot = df_global[df_global['Scale_Factor'] != 'Float (Original)'].copy()

    plt.plot(
        df_plot['Scale_Factor'], 
        df_plot['Accuracy'] * 100.0, 
        marker='o', 
        markersize=9, 
        linewidth=2.8, 
        color='#c0392b', 
        label='Global Uniform Quantization Curve'
    )
    
    plt.axhline(
        y=float_acc, 
        color='#2c3e50', 
        linestyle='--', 
        linewidth=2.0, 
        label=f'Float Model Reference (%{float_acc:.2f})'
    )

    plt.axhline(
        y=adapt_acc, 
        color='#27ae60', 
        linestyle=':', 
        linewidth=2.2, 
        label=f'Rule-Based Adaptive Model (%{adapt_acc:.2f})'
    )

    for _, row in df_plot.iterrows():
        plt.text(
            row['Scale_Factor'], 
            (row['Accuracy'] * 100.0) + 0.35, 
            f"%{row['Accuracy']*100:.1f}", 
            ha='center', 
            va='bottom', 
            fontsize=11, 
            fontweight='bold', 
            color='#7f1d1d'
        )

    plt.title(
        "A) Global Scale-Factor ($\\lambda$) Sensitivity Curve\n"
        "(Asymptotic Convergence of Test Accuracy under Uniform Scaling)",
        fontsize=16, 
        fontweight='bold', 
        pad=22
    )
    plt.xlabel("Global Scale Factor ($\\lambda$)", fontsize=14, fontweight='bold', labelpad=12)
    plt.ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold', labelpad=12)
    plt.xticks(fontsize=13, fontweight='bold')
    plt.yticks(fontsize=13)
    plt.ylim(min(df_plot['Accuracy'] * 100.0) - 3.0, float_acc + 3.0)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12, loc='lower right', frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # FIGURE 2: METHOD COMPARISON (ACCURACY AND MAX SCORE RANGE) (separate figure)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(11, 7.5))
    sns.set_style("whitegrid")

    methods = df_adapt_summary['Method'].tolist()
    accuracies = df_adapt_summary['Accuracy (%)'].tolist()
    colors = ['#2980b9', '#27ae60']

    bars = plt.bar(methods, accuracies, color=colors, edgecolor='black', width=0.45, alpha=0.88)

    for bar, (_, row) in zip(bars, df_adapt_summary.iterrows()):
        h = bar.get_height()
        aralik_str = f"[{row['Min Integer Score']}, {row['Max Integer Score']}]" if row['Min Integer Score'] != 'N/A' else 'Original Continuous'
        label_text = f"%{row['Accuracy (%)']:.2f}\n(Score Band: {aralik_str})"
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            h * 0.88, 
            label_text, 
            ha='center', 
            va='center', 
            fontsize=12, 
            fontweight='bold', 
            color='white'
        )

    plt.title(
        "B) Original Float Model vs. Rule-Based Adaptive Scorecard\n"
        "(Human-Verifiable Integer Score Conversion with Near-Zero Accuracy Loss)",
        fontsize=16, 
        fontweight='bold', 
        pad=22
    )
    plt.ylabel("Test Accuracy (%)", fontsize=14, fontweight='bold', labelpad=12)
    plt.xticks(fontsize=13, fontweight='bold')
    plt.yticks(fontsize=13)
    plt.ylim(0, max(accuracies) * 1.18)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # FIGURE 3: PER-RULE OPTIMAL LAMBDA DISTRIBUTION (dynamic binning)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 7.5))
    sns.set_style("whitegrid")

    scales = df_rule_scales['Optimal_Scale_Factor']
    unique_scale_count = scales.nunique()

    # If there are few unique values (<= 15), draw individual bars.
    # Otherwise bin the coefficients into logical ranges to keep the plot readable.
    if unique_scale_count <= 15:
        scale_counts = scales.value_counts().sort_index()
        x_labels = [f"λ={int(s)}" for s in scale_counts.index]
        y_values = scale_counts.values
        x_axis_title = "Selected Local Scale Factor ($\\lambda_h$)"
    else:
        # Fixed pool of logical bins
        candidate_bins = [0, 10, 25, 50, 100, 250, 500, 1000]
        # Keep only bin edges below max_adaptive_scale, then append the ceiling
        bins = [b for b in candidate_bins if b < max_adaptive_scale] + [max_adaptive_scale]
        bins = sorted(list(set(bins)))
        
        labels = [f"λ: {bins[i]+1}-{bins[i+1]}" if bins[i] > 0 else f"λ: 1-{bins[i+1]}" for i in range(len(bins)-1)]
        
        binned = pd.cut(scales, bins=bins, labels=labels, include_lowest=True)
        bin_counts = binned.value_counts().reindex(labels, fill_value=0)
        
        x_labels = bin_counts.index.tolist()
        y_values = bin_counts.values
        x_axis_title = r"Local Scale-Factor Ranges ($\lambda_h$ Band)"

    colors3 = sns.color_palette("Purples_r", n_colors=len(x_labels))
    bars3 = plt.bar(x_labels, y_values, color=colors3, edgecolor='black', width=0.55, alpha=0.88)

    max_c = max(y_values) if len(y_values) > 0 else 1
    total_rules = len(scales)

    for bar in bars3:
        h = bar.get_height()
        pct = (h / total_rules) * 100.0 if total_rules > 0 else 0
        if h > 0:
            label_text = f"{int(h)} rules\n(%{pct:.1f})"
        else:
            label_text = "0"
            
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            h + (max_c * 0.025), 
            label_text, 
            ha='center', 
            va='bottom', 
            fontsize=11.5, 
            fontweight='bold', 
            color='#4a235a'
        )

    plt.title(
        "C) Distribution of Optimal Local Scale Factors\n"
        f"(Total H={total_rules} rules | median $\\lambda$={scales.median():.1f} | tolerance <= %{tolerance_error*100:.0f})",
        fontsize=16, 
        fontweight='bold', 
        pad=22
    )
    plt.xlabel(x_axis_title, fontsize=14, fontweight='bold', labelpad=12)
    plt.ylabel("Rule Count", fontsize=14, fontweight='bold', labelpad=12)
    plt.xticks(fontsize=12, fontweight='bold', rotation=20 if len(x_labels) > 6 else 0)
    plt.yticks(fontsize=13)
    plt.ylim(0, max_c * 1.30)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

####################################################################################
####################################################################################
####################################################################################

@torch.no_grad()
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
        
        valid_rules = (m != 0)
        v_left = (C[valid_rules] - W[valid_rules] - b[valid_rules]) / m[valid_rules]
        v_center = (C[valid_rules] - b[valid_rules]) / m[valid_rules]
        v_right = (C[valid_rules] + W[valid_rules] - b[valid_rules]) / m[valid_rules]
        
        orig_val = single_record[0, feature_idx].item()
        f_min, f_max = dataset_mins[feature_idx].item() if isinstance(dataset_mins[feature_idx], torch.Tensor) else dataset_mins[feature_idx], dataset_maxs[feature_idx].item() if isinstance(dataset_maxs[feature_idx], torch.Tensor) else dataset_maxs[feature_idx]
        
        breakpoints = torch.cat([v_left, v_center, v_right, torch.tensor([f_min, f_max, orig_val], device=device)])
        breakpoints = breakpoints[(breakpoints >= f_min) & (breakpoints <= f_max)]
        sorted_points = torch.sort(torch.unique(torch.round(breakpoints * 1e5) / 1e5)).values
        
        K = len(sorted_points)
        test_v = sorted_points.unsqueeze(1) 
        
        h_K = m.unsqueeze(0) * test_v + b.unsqueeze(0) 
        act_K = torch.relu(W - torch.abs(C - h_K)) / (W + 1e-8) 
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
    
    is_english = False
    if lang == 'en':
        is_english = True
    elif lang is None and ((title and any(w in title.lower() for w in ['recourse', 'analytic', 'shift', 'effort'])) or 
                           (xlabel and any(w in xlabel.lower() for w in ['shift', 'delta', 'unit']))):
        is_english = True

    for bar, action in zip(bars, displayed_actions):
        width = bar.get_width()
        x_pos = max(0.0, width) + offset
        
        if is_english:
            annotation = f"Current: {action['orig']:+.2f} \u2192 Target: {action['target']:+.2f} (\u0394 = {action['diff']:+.2f})"
        else:
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
        ax.set_title(f"Class {target_class} Independent Alternative Actions Sufficient for Prediction\n(Sorted from Lowest to Highest Effort)",
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


####################################################################################
####################################################################################
####################################################################################
