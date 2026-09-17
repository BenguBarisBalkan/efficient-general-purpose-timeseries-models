@echo off

call ..\..\..\..\venv\Scripts\activate.bat

rem set CUDA_VISIBLE_DEVICES=0

set model_name=TimeMixer

set seq_len=96
set e_layers=3
set down_sampling_layers=3
set down_sampling_window=2
set learning_rate=0.01
set d_model=16
set d_ff=32
set batch_size=32
set train_epochs=20
set patience=10

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_%seq_len%_96 --model %model_name% --data custom --features M --seq_len %seq_len% --label_len 0 --pred_len 96 --e_layers %e_layers% --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des "Exp" --itr 1 --d_model %d_model% --d_ff %d_ff% --batch_size %batch_size% --learning_rate %learning_rate% --train_epochs %train_epochs% --patience %patience% --down_sampling_layers %down_sampling_layers% --down_sampling_method avg --down_sampling_window %down_sampling_window%

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_%seq_len%_192 --model %model_name% --data custom --features M --seq_len %seq_len% --label_len 0 --pred_len 192 --e_layers %e_layers% --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des "Exp" --itr 1 --d_model %d_model% --d_ff %d_ff% --batch_size %batch_size% --learning_rate %learning_rate% --train_epochs %train_epochs% --patience %patience% --down_sampling_layers %down_sampling_layers% --down_sampling_method avg --down_sampling_window %down_sampling_window%

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_%seq_len%_336 --model %model_name% --data custom --features M --seq_len %seq_len% --label_len 0 --pred_len 336 --e_layers %e_layers% --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des "Exp" --itr 1 --d_model %d_model% --d_ff %d_ff% --batch_size %batch_size% --learning_rate %learning_rate% --train_epochs %train_epochs% --patience %patience% --down_sampling_layers %down_sampling_layers% --down_sampling_method avg --down_sampling_window %down_sampling_window%

python -u run.py --task_name long_term_forecast --is_training 1 --root_path ./dataset/electricity/ --data_path electricity.csv --model_id ECL_%seq_len%_720 --model %model_name% --data custom --features M --seq_len %seq_len% --label_len 0 --pred_len 720 --e_layers %e_layers% --d_layers 1 --factor 3 --enc_in 321 --dec_in 321 --c_out 321 --des "Exp" --itr 1 --d_model %d_model% --d_ff %d_ff% --batch_size %batch_size% --learning_rate %learning_rate% --train_epochs %train_epochs% --patience %patience% --down_sampling_layers %down_sampling_layers% --down_sampling_method avg --down_sampling_window %down_sampling_window%