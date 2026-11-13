import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from .rules import WEIGHT_THRESHOLD

# --- HELPER FUNCTION ---
def get_lr(optimizer):
    """Return the current learning rate in the optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


# --- TRAIN CYCLE THAT RESTORES THE BEST EPOCH WEIGHTS ---
def train_cycle(
    model, 
    train_loader, 
    val_loader,
    device, 
    epochs=250, 
    patience=20, 
    lr=0.001, 
    min_lr=1e-6,
    lr_patience=10,
    lr_factor=0.5,
    l1_lambda_feature=0.0005,
    l1_lambda_output=0.0005,
    max_norm=1.0,
    phase_name="Training",
    verbose=True,
    log_interval=1,
    show_l1_loss=False
):
    """
    Run one training/fine-tuning cycle for a SCORENet model.

    Applies layer-wise L1 sparsity regularization, gradient clipping, and
    plateau-based learning-rate reduction (ReduceLROnPlateau). When early
    stopping triggers, the weights with the highest validation accuracy are
    restored.

    IMPORTANT: Early stopping, the learning-rate schedule, and best-weight
    selection use only ``val_loader``. The test set is never used for model
    selection; evaluate it once, after training is complete.

    Args:
        model (SCORENet): Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader (required).
        device: 'cuda' or 'cpu'.
        epochs (int): Maximum number of epochs.
        patience (int): Early-stopping patience.
        lr (float): Initial learning rate.
        min_lr (float): Lower bound of the scheduler's learning rate.
        lr_patience (int): Epochs to wait before reducing the LR.
        lr_factor (float): LR reduction factor (e.g. 0.5 halves it).
        l1_lambda_feature (float): L1 sparsity coefficient for the W_f layer.
        l1_lambda_output (float): L1 sparsity coefficient for the W_o layer.
        max_norm (float): Gradient-clipping ceiling to prevent explosion.
        phase_name (str): Phase name used in console output.
        verbose (bool): Whether to log per-epoch output.
        log_interval (int): Number of epochs between log lines.

    Returns:
        model (SCORENet): Model loaded with this cycle's best weights.
        best_cycle_acc (float): Highest validation accuracy in this cycle.
    """
    if val_loader is None:
        raise ValueError(
            "val_loader is required. Using the test set as a validation set "
            "causes data leakage and optimistically biased results. Create a "
            "separate validation set with val_size>0 in the data loader."
        )

    if verbose:
        print(f"\n{'='*15} {phase_name} STARTING {'='*15}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=lr_patience, factor=lr_factor, min_lr=min_lr
    )

    best_cycle_acc = -1.0
    best_cycle_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_correct = 0
        total_train_samples = 0
        running_ce, running_l1_feature, running_l1_output = 0.0, 0.0, 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            ce_loss = F.cross_entropy(logits, batch_y)

            # Layer-wise L1 sparsity
            l1_feat = model.feature_linear.weight.abs().sum() if l1_lambda_feature > 0 else torch.tensor(0.0, device=device)
            l1_out = model.output_linear.weight.abs().sum() if l1_lambda_output > 0 else torch.tensor(0.0, device=device)

            total_loss = ce_loss + (l1_lambda_feature * l1_feat) + (l1_lambda_output * l1_out)

            optimizer.zero_grad()
            total_loss.backward()

            # Prevent gradient explosion
            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            optimizer.step()

            running_ce += ce_loss.item()
            running_l1_feature += (l1_lambda_feature * l1_feat).item()
            running_l1_output += (l1_lambda_output * l1_out).item()
            train_correct += (logits.argmax(1) == batch_y).sum().item()
            total_train_samples += batch_y.size(0)

        # Validation phase (not the test set)
        model.eval()
        val_loss, val_correct, total_val_samples = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                v_logits = model(batch_x)
                val_loss += F.cross_entropy(v_logits, batch_y).item()
                val_correct += (v_logits.argmax(1) == batch_y).sum().item()
                total_val_samples += batch_y.size(0)

        if total_val_samples == 0:
            raise ValueError("val_loader returned no samples; the validation set is empty.")

        val_acc = val_correct / total_val_samples
        train_acc = train_correct / max(total_train_samples, 1)
        
        # Update learning-rate scheduler
        scheduler.step(val_acc)

        # Track and back up the best model in this cycle
        if val_acc > best_cycle_acc:
            best_cycle_acc = val_acc
            best_cycle_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and ((epoch + 1) % log_interval == 0 or epoch == epochs - 1):
            if show_l1_loss:
                print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {get_lr(optimizer):.6f} | L1 Feature:{running_l1_feature:.6f} |  L1 Output:{running_l1_output:.6f} ")
            else:
                print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {get_lr(optimizer):.6f}")

        if patience_counter >= patience:
            if verbose:
                print(f"[!] Early stopping triggered ({patience} epochs without improvement). Best in cycle: {best_cycle_acc:.4f}")
            break

    # Restore this cycle's best weights when the loop ends
    if best_cycle_weights is not None:
        model.load_state_dict(best_cycle_weights)

    return model, best_cycle_acc


# --- ITERATIVE PRUNING AND MODEL MANAGER ---
def run_iterative_pruning_pipeline(
    model, 
    train_loader, 
    val_loader,
    device,
    base_epochs=250,
    base_patience=20,
    base_lr=0.001,
    prune_epochs=250,
    prune_patience=20,
    prune_lr=0.001,
    num_prune_iters=9,
    prune_amount_feature=0.20,
    prune_amount_output=0.0,
    l1_lambda_feature=0.0005,
    l1_lambda_output=0.0005,
    lr_patience=10,
    lr_factor=0.5,
    max_norm=1.0,
    verbose=True,
    show_l1_loss=False,
    history_path=None,
    selection="strict",
    smoothing_window=3,
):
    """
    Manage end-to-end iterative magnitude pruning (IMP) and fine-tuning for a
    SCORENet model.

    Stages:
      1. Base model training: train from scratch and record the initial best.
      2. Iterative pruning chain: prune the specified layers each iteration and
         retrain. The intermediate model with the highest validation accuracy
         (Global Best) is frozen and preserved.
      3. Mask consolidation: remove PyTorch pruning hooks to make the weights
         permanent.

    Args:
        model (SCORENet): Initial architecture.
        train_loader: Training data loader.
        val_loader: Validation data loader (required). Model selection and
            early stopping use only this set; the test set must never be
            passed to this function.
        device: 'cuda' or 'cpu'.
        base_epochs (int): Epochs in the initial (base) training.
        base_patience (int): Early-stopping patience for the initial training.
        base_lr (float): Learning rate for the initial training.
        prune_epochs (int): Retraining epochs per pruning iteration.
        prune_patience (int): Early-stopping patience per pruning iteration.
        prune_lr (float): Retraining learning rate after pruning.
        num_prune_iters (int): Total number of pruning iterations.
        prune_amount_feature (float): Fraction of W_f weights pruned per step (e.g. 0.20 -> 20%).
        prune_amount_output (float): Fraction of W_o weights pruned per step.
        l1_lambda_feature (float): L1 regularization coefficient for W_f.
        l1_lambda_output (float): L1 regularization coefficient for W_o.
        lr_patience (int): Scheduler patience.
        lr_factor (float): Scheduler LR reduction factor.
        max_norm (float): Gradient-clipping ceiling.
        verbose (bool): Whether to print process logs.
        history_path (str | None): If given, write each iteration's feature
            sparsity and validation accuracy to this JSON file. Figure
            notebooks read pruning trajectories from it.
        selection (str): Selection rule for the "Global Best" model.
            ``"strict"`` (default) uses only validation accuracy and keeps the
            first highest value. ``"compact"`` prefers the model with fewer
            active rules at equal accuracy. ``"smoothed"`` selects using a
            ``smoothing_window``-iteration moving average, avoiding selection
            of a single lucky spike on small validation sets.
        smoothing_window (int): Window size for ``selection="smoothed"``.

    Returns:
        global_best_model (SCORENet): Model with the highest validation accuracy over all iterations.
        final_iter_model (SCORENet): Final-iteration model with the highest sparsity.
    """
    if selection not in {"strict", "compact", "smoothed"}:
        raise ValueError(
            f"Bilinmeyen selection={selection!r}. "
            "Options: 'strict', 'compact', 'smoothed'."
        )

    global_best_acc = -1.0
    global_best_model = None
    global_best_rules = None
    iteration_models = []  # (iteration, val_acc, active_rules, model_copy)

    # Training history: figure notebooks read this file. The test set is never
    # used here; the recorded accuracy is validation accuracy.
    history = {
        "schema_version": 1,
        "H_init": int(model.H),
        "num_prune_iters": int(num_prune_iters),
        "prune_amount_feature": float(prune_amount_feature),
        "prune_amount_output": float(prune_amount_output),
        "iterations": [],
    }

    def _record_iteration(iteration, phase, val_acc):
        """Record an iteration's sparsity and validation accuracy."""
        weights = model.feature_linear.weight.detach()
        total = max(int(weights.numel()), 1)
        zero_count = int((weights == 0).sum().item())
        active_rules = int((weights.abs().sum(dim=1) > WEIGHT_THRESHOLD).sum().item())
        history["iterations"].append({
            "iteration": int(iteration),
            "phase": phase,
            "feature_sparsity_pct": 100.0 * zero_count / total,
            "val_accuracy": float(val_acc),
            "active_rules": active_rules,
        })
        # Smoothed selection needs the model itself after the loop.
        if selection == "smoothed":
            iteration_models.append(
                (int(iteration), float(val_acc), active_rules, copy.deepcopy(model))
            )

    def _active_rule_count(candidate_model):
        """Number of rules with a non-zero W_f row (active rules)."""
        weights = candidate_model.feature_linear.weight.detach()
        return int((weights.abs().sum(dim=1) > WEIGHT_THRESHOLD).sum().item())

    def _update_global_best(iteration, val_acc, candidate_model):
        """Update the Global Best model according to the selection rule.

        For ``selection="smoothed"``, the decision is made after the loop using
        the moving average, so no update happens here.
        """
        nonlocal global_best_acc, global_best_model, global_best_rules
        rules = _active_rule_count(candidate_model)

        if selection == "strict":
            better = val_acc > global_best_acc
        elif selection == "compact":
            better = val_acc > global_best_acc or (
                val_acc == global_best_acc
                and (global_best_rules is None or rules < global_best_rules)
            )
        else:  # smoothed
            better = False

        if better:
            global_best_acc = float(val_acc)
            global_best_model = copy.deepcopy(candidate_model)
            global_best_rules = rules
            history["global_best_iteration"] = int(iteration)
            if verbose:
                print(
                    f"--> [NEW RECORD] Global best model updated! "
                    f"(Iteration {iteration} | Val Acc: {global_best_acc:.4f} | "
                    f"Rules: {rules})"
                )

    # ----------------------------------------------------
    # STAGE 1: Base model training
    # ----------------------------------------------------
    model, base_best_acc = train_cycle(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        device=device, 
        epochs=base_epochs, 
        patience=base_patience, 
        lr=base_lr, 
        lr_patience=lr_patience,
        lr_factor=lr_factor,
        l1_lambda_feature=l1_lambda_feature,
        l1_lambda_output=l1_lambda_output,
        max_norm=max_norm,
        phase_name="INITIAL TRAINING (BASE MODEL)",
        verbose=verbose,
        show_l1_loss=show_l1_loss
    )

    # Record the base-model result as the global best
    _record_iteration(0, "base", base_best_acc)
    history["global_best_iteration"] = 0
    _update_global_best(0, base_best_acc, model)

    # ----------------------------------------------------
    # STAGE 2: Iterative pruning chain
    # ----------------------------------------------------
    for i in range(num_prune_iters):
        # 1. Apply layer-wise pruning
        if prune_amount_feature > 0:
            prune.l1_unstructured(model.feature_linear, name='weight', amount=prune_amount_feature)
        if prune_amount_output > 0:
            prune.l1_unstructured(model.output_linear, name='weight', amount=prune_amount_output)

        # Compute sparsity ratios
        f_zeros = float(torch.sum(model.feature_linear.weight == 0)) / float(model.feature_linear.weight.nelement())
        o_zeros = float(torch.sum(model.output_linear.weight == 0)) / float(model.output_linear.weight.nelement())
        
        if verbose:
            print(f"\n[PRUNING] Iteration {i+1}/{num_prune_iters}: Feature_Linear zeros: %{f_zeros*100:.1f} | Output_Linear zeros: %{o_zeros*100:.1f}")

        # 2. Retrain the model (fine-tuning)
        model, iter_best_acc = train_cycle(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader,
            device=device, 
            epochs=prune_epochs, 
            patience=prune_patience, 
            lr=prune_lr, 
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            l1_lambda_feature=l1_lambda_feature,
            l1_lambda_output=l1_lambda_output,
            max_norm=max_norm,
            phase_name=f"RETRAINING (Iteration {i+1}/{num_prune_iters})",
            verbose=verbose,
            show_l1_loss=show_l1_loss
        )

        # 3. Update the Global Best model if a new accuracy record is set
        _record_iteration(i + 1, "prune", iter_best_acc)
        _update_global_best(i + 1, iter_best_acc, model)

    # ----------------------------------------------------
    # STAGE 2b: Smoothed selection
    # ----------------------------------------------------
    # Instead of selecting a single lucky spike on small validation sets, choose
    # the model with the highest ``smoothing_window``-iteration moving average.
    # Ties prefer the model with fewer active rules.
    if selection == "smoothed":
        if not iteration_models:
            raise RuntimeError("iteration models were not saved for smoothed selection.")

        accs = [record[1] for record in iteration_models]
        window = max(1, int(smoothing_window))
        scores = []
        for index in range(len(accs)):
            low = max(0, index - window // 2)
            high = min(len(accs), index + window // 2 + 1)
            scores.append(sum(accs[low:high]) / (high - low))

        best_score = max(scores)
        candidates = [i for i, score in enumerate(scores) if score == best_score]
        best_index = min(candidates, key=lambda i: iteration_models[i][2])

        iteration, val_acc, rules, best_model = iteration_models[best_index]
        global_best_acc = float(val_acc)
        global_best_model = best_model
        global_best_rules = int(rules)
        history["global_best_iteration"] = int(iteration)
        history["smoothed_score"] = float(best_score)
        history["selection"] = "smoothed"
        if verbose:
            print(
                f"--> [SMOOTHED SELECTION] Iteration {iteration} selected "
                f"(Val Acc: {global_best_acc:.4f} | Rules: {rules} | "
                f"Smoothed: {best_score:.4f})"
            )
    else:
        history["selection"] = selection

    # ----------------------------------------------------
    # STAGE 3: Make masks permanent (prune.remove)
    # ----------------------------------------------------
    # 1. Make the final pruned model's masks permanent
    if prune.is_pruned(model.feature_linear):
        prune.remove(model.feature_linear, 'weight')
    if prune.is_pruned(model.output_linear):
        prune.remove(model.output_linear, 'weight')
    final_iter_model = model

    # 2. Make the Global Best model's masks permanent
    if global_best_model is not None:
        if prune.is_pruned(global_best_model.feature_linear):
            prune.remove(global_best_model.feature_linear, 'weight')
        if prune.is_pruned(global_best_model.output_linear):
            prune.remove(global_best_model.output_linear, 'weight')

    if verbose:
        print(f"\n{'='*25} TRAINING COMPLETE {'='*25}")
        print(f"1. Model: best overall (Global Best Val Acc: {global_best_acc:.4f})")
        print(f"2. Model: {num_prune_iters}. best of the final iteration (Final Pruned Model)")
        print(f"{'='*70}\n")

    # ----------------------------------------------------
    # STAGE 4: Persist the training history
    # ----------------------------------------------------
    history["final_iteration"] = int(num_prune_iters)
    history["global_best_val_accuracy"] = float(global_best_acc)
    history["final_val_accuracy"] = float(history["iterations"][-1]["val_accuracy"])

    if history_path is not None:
        output_path = Path(history_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if verbose:
            print(f"[HISTORY] Training history saved: {output_path}")

    return global_best_model, final_iter_model
