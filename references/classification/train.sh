#CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py \
#    --model resnet50 --epochs 100

CUDA_VISIBLE_DEVICES=2,3 python train.py --model resnet50 --world-size=2 --batch-size=64 --use_gtensor --workers=2
#CUDA_VISIBLE_DEVICES=2,3 python train.py --model mobilenet_v2 --world-size=2 --batch-size=64 --use_gtensor --workers=0
