@echo off

call ..\..\..\..\venv\Scripts\activate.bat

set CUDA_VISIBLE_DEVICES=0

set model_name=TimeMixer

set seq_len=96
set e_layers=2
set down_sampling_layers=3
set down_sampling_window=2
set learning_rate=0.01
set d_model=16
set d_ff=32
set batch_size=16

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_%seq_len%_96 --model %model_name% --data ETTh2 --features M --seq_len %seq_len% --label_len 0 --pred_len 96 --e_layers %e_layers% --enc_in 7 --c_out 7 --des "Exp" --itr 1 --d_model %d_model% --d_ff %d_ff% --learning_rate %learning_rate% --down_sampling_layers %down_sampling_layers% --down_sampling_method avg --down_sampling_window %down_sampling_window%

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_%seq_len%_192 --model %model_name% --data ETTh2 --features M --seq_len %seq_len% --label_len 0 --pred_len 192 --e_layers %e_layers% --enc_in 7 --c_out 7 --des "Exp" --itr 1 --d_model %d_model% --d_ff %d_ff% --learning_rate %learning_rate% --down_sampling_layers %down_sampling_layers% --down_sampling_method avg --down_sampling_window %down_sampling_window%

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_%seq_len%_336 --model %model_name% --data ETTh2 --features M --seq_len %seq_len% --label_len 0 --pred_len 336 --e_layers %e_layers% --enc_in 7 --c_out 7 --des "Exp" --itr 1 --d_model %d_model% --d_ff %d_ff% --learning_rate %learning_rate% --down_sampling_layers %down_sampling_layers% --down_sampling_method avg --down_sampling_window %down_sampling_window%

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/ETT-small/ --data_path ETTh2.csv --model_id ETTh2_%seq_len%_720 --model %model_name% --data ETTh2 --features M --seq_len %seq_len% --label_len 0 --pred_len 720 --e_layers %e_layers% --enc_in 7 --c_out 7 --des "Exp" --itr 1 --d_model %d_model% --d_ff %d_ff% --learning_rate %learning_rate% --down_sampling_layers %down_sampling_layers% --down_sampling_method avg --down_sampling_window %down_sampling_window%