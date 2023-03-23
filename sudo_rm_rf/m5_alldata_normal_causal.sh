exp_name=m5_alldata_mild_causal
mkdir /home/ubuntu/Data/enric_models/$exp_name
nohup python run_microson_v1.py \
--separation_task enh_noisyreverberant --n_train 220701 --n_test 2385 --n_val 2408 \
--out_channels 512 --in_channels 256 --enc_kernel_size 21 --num_blocks 16 -cad 0 -bs 12 --divide_lr_by 3. --upsampling_depth 5 \
--patience 8 -fs 16000 -tags $exp_name --project_name microson_v1 --zero_pad --clip_grad_norm 5.0 --normalize_online --online_mix \
--checkpoints_path /home/ubuntu/Data/enric_models/$exp_name --save_checkpoint_every 1 --n_epochs 25 --mild_target > $exp_name.txt
