#!/usr/bin/env bash
set -euo pipefail

# Dry-run by default. To actually move files:
#   APPLY=1 bash repo_cleanup_move_plan.sh
APPLY=${APPLY:-0}

run() {
  if [[ "$APPLY" == "1" ]]; then
    echo "+ $*"
    eval "$@"
  else
    echo "DRY-RUN: $*"
  fi
}

move_file() {
  src="$1"
  dst="$2"
  if [[ -e "$src" ]]; then
    run "mkdir -p \"$(dirname "$dst")\""
    run "mv \"$src\" \"$dst\""
  else
    echo "SKIP missing: $src"
  fi
}

# Create target folders
run "mkdir -p scripts/data_prep scripts/pretrain scripts/finetune scripts/benchmark scripts/plot"
run "mkdir -p scripts/archive/validation scripts/archive/legacy_pbm scripts/archive/legacy_pretrain scripts/archive/legacy_data_prep scripts/archive/legacy_htselex scripts/archive/legacy_plots scripts/archive/_trash_candidates scripts/archive/review"

move_file "scripts/.DS_Store" "scripts/archive/_trash_candidates/.DS_Store"
move_file "scripts/.ipynb_checkpoints/finetune_pbm_max-checkpoint.py" "scripts/archive/_trash_candidates/finetune_pbm_max-checkpoint.py"
move_file "scripts/__pycache__/pretrain_bendability_stage1.cpython-313.pyc" "scripts/archive/_trash_candidates/pretrain_bendability_stage1.cpython-313.pyc"
move_file "scripts/add_htselex_1mer12flex.py" "scripts/benchmark/add_htselex_1mer12flex.py"
move_file "scripts/baseline_pbm_ridge_1mer.py" "scripts/finetune/baseline_pbm_ridge_1mer.py"
move_file "scripts/benchmark_gcpbm_four_checkpoints.py" "scripts/benchmark/benchmark_gcpbm_four_checkpoints.py"
move_file "scripts/benchmark_htselex_option2_panelA.py" "scripts/benchmark/benchmark_htselex_option2_panelA.py"
move_file "scripts/benchmark_htselex_option2_pilot.py" "scripts/archive/legacy_htselex/benchmark_htselex_option2_pilot.py"
move_file "scripts/benchmark_htselex_panelA.py" "scripts/archive/legacy_htselex/benchmark_htselex_panelA.py"
move_file "scripts/compute_flex_norm_stats.py" "scripts/data_prep/compute_flex_norm_stats.py"
move_file "scripts/download_hg38_ucsc.sh" "scripts/data_prep/download_hg38_ucsc.sh"
move_file "scripts/eval_pbm_r2_flex_maxpool.py" "scripts/archive/legacy_pbm/eval_pbm_r2_flex_maxpool.py"
move_file "scripts/finetune_pbm_hidden_plus_flex_poslinear.py" "scripts/finetune/finetune_pbm_hidden_plus_flex_poslinear.py"
move_file "scripts/finetune_pbm_hidden_poslinear.py" "scripts/finetune/finetune_pbm_hidden_poslinear.py"
move_file "scripts/finetune_pbm_max.py" "scripts/archive/legacy_pbm/finetune_pbm_max.py"
move_file "scripts/finetune_pbm_max_baseline_random_encoder.py" "scripts/archive/legacy_pbm/finetune_pbm_max_baseline_random_encoder.py"
move_file "scripts/finetune_pbm_max_baseline_random_encoder_maxpool.py" "scripts/archive/legacy_pbm/finetune_pbm_max_baseline_random_encoder_maxpool.py"
move_file "scripts/finetune_pbm_max_flex_maxpool_betterhead.py" "scripts/archive/legacy_pbm/finetune_pbm_max_flex_maxpool_betterhead.py"
move_file "scripts/finetune_pbm_max_flex_maxpool_betterhead.py.bak" "scripts/archive/_trash_candidates/finetune_pbm_max_flex_maxpool_betterhead.py.bak"
move_file "scripts/finetune_pbm_max_hidden_maxpool.py" "scripts/archive/legacy_pbm/finetune_pbm_max_hidden_maxpool.py"
move_file "scripts/finetune_pbm_max_unfreeze_last.py" "scripts/archive/legacy_pbm/finetune_pbm_max_unfreeze_last.py"
move_file "scripts/make_hg38_windows_256.py" "scripts/archive/legacy_data_prep/make_hg38_windows_256.py"
move_file "scripts/make_hg38_windows_256_50k.py" "scripts/archive/legacy_data_prep/make_hg38_windows_256_50k.py"
move_file "scripts/make_windows_from_yaml.py" "scripts/data_prep/make_windows_from_yaml.py"
move_file "scripts/panelC_gcpbm_sample_efficiency.py" "scripts/benchmark/panelC_gcpbm_sample_efficiency.py"
move_file "scripts/panel_c_ridge_on_transformer_features.py" "scripts/archive/legacy_plots/panel_c_ridge_on_transformer_features.py"
move_file "scripts/plot_gcpbm_4model_boxplot.py" "scripts/plot/plot_gcpbm_4model_boxplot.py"
move_file "scripts/plot_panelA0_random_vs_flexonly_rawR2.py" "scripts/archive/legacy_plots/plot_panelA0_random_vs_flexonly_rawR2.py"
move_file "scripts/plot_panelA1_ridge_vs_flexonly_rawR2.py" "scripts/plot/plot_panelA1_ridge_vs_flexonly_rawR2.py"
move_file "scripts/plot_panelA2_ridge_vs_hiddenonly_rawR2.py" "scripts/plot/plot_panelA2_ridge_vs_hiddenonly_rawR2.py"
move_file "scripts/plot_panelA3_hidden_vs_hiddenplusflex_rawR2.py" "scripts/plot/plot_panelA3_hidden_vs_hiddenplusflex_rawR2.py"
move_file "scripts/plot_panelA_r2_ridge_vs_transformer.py" "scripts/archive/legacy_plots/plot_panelA_r2_ridge_vs_transformer.py"
move_file "scripts/plot_panelB_pred_vs_obs.py" "scripts/archive/legacy_plots/plot_panelB_pred_vs_obs.py"
move_file "scripts/plot_panelB_seed2_ridge_vs_hiddenplusflex.py" "scripts/plot/plot_panelB_seed2_ridge_vs_hiddenplusflex.py"
move_file "scripts/plot_panelC_sample_size_vs_r2.py" "scripts/archive/legacy_plots/plot_panelC_sample_size_vs_r2.py"
move_file "scripts/plot_panelC_three_lines.py" "scripts/archive/legacy_plots/plot_panelC_three_lines.py"
move_file "scripts/plot_panelC_three_lines_pct.py" "scripts/archive/legacy_plots/plot_panelC_three_lines_pct.py"
move_file "scripts/pretrain_bendability_stage1.py" "scripts/pretrain/pretrain_bendability_stage1.py"
move_file "scripts/pretrain_bendability_stage2_mlm.py" "scripts/pretrain/pretrain_bendability_stage2_mlm.py"
move_file "scripts/pretrain_hg38_tiny.py" "scripts/archive/legacy_pretrain/pretrain_hg38_tiny.py"
move_file "scripts/pretrain_hg38_tiny_trainval.py" "scripts/pretrain/pretrain_hg38_tiny_trainval.py"
move_file "scripts/replot_htselex_2x2_clean.py" "scripts/plot/replot_htselex_2x2_clean.py"
move_file "scripts/replot_htselex_bendflex_vs_baselines.py" "scripts/archive/legacy_plots/replot_htselex_bendflex_vs_baselines.py"
move_file "scripts/replot_htselex_option2_clean.py" "scripts/archive/legacy_plots/replot_htselex_option2_clean.py"
move_file "scripts/replot_panelA1_clean.py" "scripts/archive/legacy_plots/replot_panelA1_clean.py"
move_file "scripts/replot_panelC_gcpbm_no_errorbar.py" "scripts/plot/replot_panelC_gcpbm_no_errorbar.py"
move_file "scripts/split_windows_from_yaml.py" "scripts/data_prep/split_windows_from_yaml.py"
move_file "scripts/split_windows_train_val.py" "scripts/archive/legacy_data_prep/split_windows_train_val.py"
move_file "scripts/validate_all_dinuc_targets_normalized.py" "scripts/archive/validation/validate_all_dinuc_targets_normalized.py"
move_file "scripts/validate_all_dinuc_targets_normalized_loop.py" "scripts/archive/validation/validate_all_dinuc_targets_normalized_loop.py"
move_file "scripts/validate_batched_mixed_di_tri.py" "scripts/archive/validation/validate_batched_mixed_di_tri.py"
move_file "scripts/validate_batched_mixed_di_tri_loop.py" "scripts/archive/validation/validate_batched_mixed_di_tri_loop.py"
move_file "scripts/validate_dataloader_epoch_loop.py" "scripts/archive/validation/validate_dataloader_epoch_loop.py"
move_file "scripts/validate_dataloader_training_loop.py" "scripts/archive/validation/validate_dataloader_training_loop.py"
move_file "scripts/validate_dataloader_training_step.py" "scripts/archive/validation/validate_dataloader_training_step.py"
move_file "scripts/validate_flex_only_train_val_epoch_loop.py" "scripts/archive/validation/validate_flex_only_train_val_epoch_loop.py"
move_file "scripts/validate_four_real_targets.py" "scripts/archive/validation/validate_four_real_targets.py"
move_file "scripts/validate_four_real_targets_normalized.py" "scripts/archive/validation/validate_four_real_targets_normalized.py"
move_file "scripts/validate_four_real_targets_normalized_loop.py" "scripts/archive/validation/validate_four_real_targets_normalized_loop.py"
move_file "scripts/validate_larger_train_val_epoch_loop.py" "scripts/archive/validation/validate_larger_train_val_epoch_loop.py"
move_file "scripts/validate_mixed_di_tri_targets.py" "scripts/archive/validation/validate_mixed_di_tri_targets.py"
move_file "scripts/validate_mixed_di_tri_targets_loop.py" "scripts/archive/validation/validate_mixed_di_tri_targets_loop.py"
move_file "scripts/validate_onehot_multitask.py" "scripts/archive/validation/validate_onehot_multitask.py"
move_file "scripts/validate_padded_batched_mixed_di_tri.py" "scripts/archive/validation/validate_padded_batched_mixed_di_tri.py"
move_file "scripts/validate_padded_batched_mixed_di_tri_loop.py" "scripts/archive/validation/validate_padded_batched_mixed_di_tri_loop.py"
move_file "scripts/validate_real_twistdisp.py" "scripts/archive/validation/validate_real_twistdisp.py"
move_file "scripts/validate_real_twistdisp_loop.py" "scripts/archive/validation/validate_real_twistdisp_loop.py"
move_file "scripts/validate_small_mlm_weight_train_val_epoch_loop.py" "scripts/archive/validation/validate_small_mlm_weight_train_val_epoch_loop.py"
move_file "scripts/validate_smoke.py" "scripts/archive/validation/validate_smoke.py"
move_file "scripts/validate_train_val_epoch_loop.py" "scripts/archive/validation/validate_train_val_epoch_loop.py"
move_file "scripts/validate_two_real_targets.py" "scripts/archive/validation/validate_two_real_targets.py"
move_file "scripts/validate_two_real_targets_loop.py" "scripts/archive/validation/validate_two_real_targets_loop.py"

# Keep src/, configs/, README.md, requirements.txt, and setup.py in place for now.
# Recommended after applying: run a few smoke tests and then update README paths.
