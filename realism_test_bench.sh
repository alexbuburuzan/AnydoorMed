#!/bin/bash --login
#$ -cwd
#$ -l a100=1
#$ -pe smp.pe 12

nvidia-smi
conda activate mobi

# Define base directory for results
RESULTS_BASE_DIR="models/final_results"

# Define experiment types
# experiments=("vanilla_inpainting" "lr_ablations")
experiments=("lr_ablations_other")

# Define anomaly types
anomaly_types=("insert" "replace")

# Initialize result table function
initialize_results_table() {
    local exp_type="$1"
    local anomaly_type="$2"
    local results_table="${RESULTS_BASE_DIR}/exp_${exp_type}/realism_table_${anomaly_type}.csv"
    mkdir -p "$(dirname "${results_table}")"
    if [ ! -f "${results_table}" ]; then
        echo "Experiment,Epoch,FID,LPIPS,CLIP,DINOv2" > "${results_table}"
    fi
    echo "${results_table}"
}

# Compute scores function
compute_scores() {
    local out_dir="$1"
    local exp_name="$2"
    local epoch="$3"
    local results_table="$4"
    
    echo "Computing scores for ${exp_name} epoch ${epoch}"
    
    # FID score
    echo python eval_tool/fid_score.py --path_target "${out_dir}/pred/patch_target" --path_pred "${out_dir}/pred/patch_pred" 
    FID_SCORE=$(python eval_tool/fid_score.py --path_target "${out_dir}/pred/patch_target" --path_pred "${out_dir}/pred/patch_pred" | grep -oP 'FID:\s*\K[0-9.]+' | xargs printf "%.4f")
    # LPIPS score
    echo python eval_tool/lpips_score.py --path_target "${out_dir}/pred/patch_target" --path_pred "${out_dir}/pred/patch_pred" 
    LPIPS_SCORE=$(python eval_tool/lpips_score.py --path_target "${out_dir}/pred/patch_target" --path_pred "${out_dir}/pred/patch_pred" | grep -oP 'LPIPS:\s*\K[0-9.]+' | xargs printf "%.4f")
    
    # Get both CLIP and DINOv2 scores
    echo python eval_tool/ref_score.py --path_ref "${out_dir}/pred/anomaly_ref_resized" --path_pred "${out_dir}/pred/pred_reference"
    REF_SCORES=$(python eval_tool/ref_score.py --path_ref "${out_dir}/pred/anomaly_ref_resized" --path_pred "${out_dir}/pred/pred_reference")
    CLIP_SCORE=$(echo "$REF_SCORES" | grep -oP 'CLIP:\s*\K[0-9.]+' | xargs printf "%.4f")
    DINO_SCORE=$(echo "$REF_SCORES" | grep -oP 'DINOv2:\s*\K[0-9.]+' | xargs printf "%.4f")
    
    echo "${exp_name},${epoch},${FID_SCORE},${LPIPS_SCORE},${CLIP_SCORE},${DINO_SCORE}" >> "${results_table}"
}

# Loop through each experiment type
for exp_type in "${experiments[@]}"; do
    # Loop through each anomaly type
    for anomaly_type in "${anomaly_types[@]}"; do
        # Initialize results table for this experiment type and anomaly type
        results_table=$(initialize_results_table "${exp_type}" "${anomaly_type}")
        
        # Loop through each run folder in the experiment directory
        for run_dir in models/AnydoorMed/${exp_type}/*; do
            # Skip if not a directory
            [ ! -d "$run_dir" ] && continue
            
            # Get the run name from the directory path
            run_name=$(basename "$run_dir")
            
            # Find all checkpoint files in the run directory
            for ckpt in ${run_dir}/checkpoints/epoch=*.ckpt; do
                # Extract epoch number from checkpoint filename
                epoch=$(basename "$ckpt" | grep -o '[0-9]\{6\}')
                
                # Define output directory
                out_dir="models/final_results/exp_${exp_type}/${anomaly_type}/${run_name}/epoch${epoch}"
                echo $ckpt
                
                # Run inference
                python3 inference_test_bench.py \
                    --outdir "${out_dir}" \
                    --config "configs/anydoor.yaml" \
                    --ckpt "$ckpt" \
                    --scale "5" \
                    --ddim_steps "50" \
                    --batch_size "32" \
                    --n_workers "12" \
                    --seed "23" \
                    data.params.test.params.anomaly_type="${anomaly_type}"
                
                # Compute scores and add to results table
                compute_scores "${out_dir}" "${exp_type}/${run_name}" "${epoch}" "${results_table}"
            done
        done
    done
done
