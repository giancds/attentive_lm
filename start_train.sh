export CUDA_VISIBLE_DEVICES=0
export MODEL_DIR=${HOME}/train_lms/ptb_single

mkdir -p $MODEL_DIR

python3 -u train_attentive_lm.py \
  --config="ptb_single" \
  --train_dir=$MODEL_DIR \
  --best_models_dir=$MODEL_DIR/best_models
