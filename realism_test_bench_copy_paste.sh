nvidia-smi
conda activate anydoor_med

RESULTS_BASE_DIR="models/evaluation_results"
experiments=("poisson_blending")
anomaly_types=("insert" "replace")

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
    local results_table="$3"
    
    echo "Computing scores for ${exp_name}"
    
    # FID
    echo python eval_tool/fid_score.py --path_target "${out_dir}/pred/patch_target" --path_pred "${out_dir}/pred/patch_pred" 
    FID_SCORE=$(python eval_tool/fid_score.py --path_target "${out_dir}/pred/patch_target" --path_pred "${out_dir}/pred/patch_pred" | grep -oP 'FID:\s*\K[0-9.]+' | xargs printf "%.4f")
    # LPIPS
    echo python eval_tool/lpips_score.py --path_target "${out_dir}/pred/patch_target" --path_pred "${out_dir}/pred/patch_pred" 
    LPIPS_SCORE=$(python eval_tool/lpips_score.py --path_target "${out_dir}/pred/patch_target" --path_pred "${out_dir}/pred/patch_pred" | grep -oP 'LPIPS:\s*\K[0-9.]+' | xargs printf "%.4f")
    
    # CLIP and DINOv2 scores
    echo python eval_tool/ref_score.py --path_ref "${out_dir}/pred/anomaly_ref_resized" --path_pred "${out_dir}/pred/pred_reference"
    REF_SCORES=$(python eval_tool/ref_score.py --path_ref "${out_dir}/pred/anomaly_ref_resized" --path_pred "${out_dir}/pred/pred_reference")
    CLIP_SCORE=$(echo "$REF_SCORES" | grep -oP 'CLIP:\s*\K[0-9.]+' | xargs printf "%.4f")
    DINO_SCORE=$(echo "$REF_SCORES" | grep -oP 'DINOv2:\s*\K[0-9.]+' | xargs printf "%.4f")
    
    echo "${exp_name},0,${FID_SCORE},${LPIPS_SCORE},${CLIP_SCORE},${DINO_SCORE}" >> "${results_table}"
}

for exp_type in "${experiments[@]}"; do
    for anomaly_type in "${anomaly_types[@]}"; do
        # Initialize results table for this experiment type and anomaly type
        results_table=$(initialize_results_table "${exp_type}" "${anomaly_type}")
        
        # Define output directory
        out_dir="models/evaluation_results/exp_${exp_type}/${anomaly_type}"
        
        # Run inference
        python3 inference_test_bench.py \
            --outdir "${out_dir}" \
            --config "configs/anydoor.yaml" \
            --scale "5" \
            --copy_paste \
            --ddim_steps "1" \
            --batch_size "32" \
            --n_workers "12" \
            --seed "23" \
            data.params.test.params.anomaly_type="${anomaly_type}"
        
        # Compute scores and add to results table
        compute_scores "${out_dir}" "${exp_type}" "${results_table}"
    done
done
