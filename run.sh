CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --config config_run/train_vae/vae_city_rgb.yaml --num_workers 8

CUDA_VISIBLE_DEVICES=2,3,4,5 python3 main.py --config config_run/train_vae/vae_city_depth.yaml --num_workers 8

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main.py --config config_run/train_fm/fm_city_rgb.yaml --num_workers 8

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py --config config_run/train_fm/fm_city_rgb.yaml --num_workers 8

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main.py --config config_run/train_fm/fm_city_rgb.yaml --num_workers 8


