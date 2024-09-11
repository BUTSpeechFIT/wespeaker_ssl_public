#!/bin/bash
#
#$ -cwd
#$ -V
#$ -N WavLM-Large
#$ -o log/Large_MHFA_$JOB_ID_$TASK_ID.out
#$ -e log/Large_MHFA_$JOB_ID_$TASK_ID.err
#$ -q long.q@supergpu8,long.q@supergpu9,long.q@supergpu10,long.q@supergpu11,long.q@supergpu12,long.q@supergpu13,long.q@supergpu14,long.q@supergpu15,long.q@supergpu16,long.q@supergpu17,long.q@supergpu18
#$ -pe smp 4
#$ -l gpu_ram=20G,ram_free=10G,mem_free=10G,gpu=1

work_dir=/mnt/matylda6/pengjy/wespeaker/wespeaker/examples/voxceleb/v4/
cd $work_dir
source ~/.bashrc
conda activate wespeaker

num_gpus=4
gpus=$(python -c "from sys import argv; from safe_gpu import safe_gpu; safe_gpu.claim_gpus(int(argv[1])); print( safe_gpu.gpu_owner.devices_taken   )" $num_gpus | sed "s: ::g")
gpus=(`echo $gpus | cut -d '[' -f2 | cut -d ']' -f1 `)


. ./path.sh || exit 1
# . ./lumi.sh || exit 1 # For AMD-ROCM user.

stage=8 #3
stop_stage=8 #6

data=data
data_type="raw"  # shard/raw

config=conf/MHFA/MHFA_Large_Frozen.yaml
ft_config=conf/MHFA/MHFA_Large_Frozen_FT.yaml 
lm_config=conf/MHFA/MHFA_Large_Frozen_FT_MLFT.yaml 

exp_dir=exp/WavLM-Large-MHFA-Head64-emb256-3s-Epoch10-Fixed


base_port=1024
max_port=40000
current_time=$(date +%s)
port=$((current_time % (max_port - base_port) + base_port))

# gpus="[0,1,3,4,5,6,7]"
# gpus="[0,1,2,3]"

num_avg=2
checkpoint=

trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

# setup for large margin fine-tuning

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 4 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in vox2_dev vox1; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 32 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
  # Convert all musan data to LMDB
  python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # Convert all rirs data to LMDB
  python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --master_addr=localhost --master_port=${port} --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train_hubert.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox2_dev/${data_type}.list \
      --train_label ${data}/vox2_dev/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      --PORT ${port} \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}
  
  avg_model=$exp_dir/models/model_3.pt
  model_path=$avg_model
  echo "Extract embeddings ..."
  local/extract_vox.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 4 --gpus $gpus --data_type $data_type --data ${data}
  local/extract_vox2.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 4 --gpus $gpus --data_type $data_type --data ${data}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score norm ..."
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Score calibration ..."
  local/score_calibration.sh \
    --stage 1 --stop-stage 5 \
    --score_norm_method $score_norm_method \
    --calibration_trial "vox2_cali.kaldi" \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Full fine-tuning ..."
  ft_exp_dir=${exp_dir}-FT
  mkdir -p ${ft_exp_dir}/models
  cp ${exp_dir}/models/avg_model.pt ${ft_exp_dir}/models/model_0.pt
  bash run_MHFA2.sh --stage 4 --stop_stage 6 \
      --data ${data} \
      --data_type ${data_type} \
      --config ${ft_config} \
      --exp_dir ${ft_exp_dir} \
      --gpus $gpus \
      --num_avg 1 \
      --checkpoint ${ft_exp_dir}/models/model_0.pt \
      --trials "$trials" \
      --score_norm_method ${score_norm_method} \
      --top_n ${top_n}
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Large Margin Fine-tuning ..."
  lm_exp_dir=${exp_dir}-FT-LMFT
  mkdir -p ${lm_exp_dir}/models
  cp ${exp_dir}/models/avg_model.pt ${lm_exp_dir}/models/model_0.pt
  bash run_MHFA.sh --stage 3 --stop_stage 7 \
      --data ${data} \
      --data_type ${data_type} \
      --config ${lm_config} \
      --exp_dir ${lm_exp_dir} \
      --gpus $gpus \
      --num_avg 1 \
      --checkpoint ${lm_exp_dir}/models/model_0.pt \
      --trials "$trials" \
      --score_norm_method ${score_norm_method} \
      --top_n ${top_n}
fi
