# python3 main.py \
#     --all_clients \
#     --fed fedavg \
#     --gpu 0 \
#     --seed 3 \
#     --sampling iid \
#     --num_channels 3 \
#     --dataset cifar

python3 main.py \
    --all_clients \
    --fed fedavg \
    --gpu 0 \
    --debug \
    --seed 3 \
    --sampling iid \
    --num_channels 1 \
    --num_classes 2 \
    --num_users 5 \
    --dataset FairFace