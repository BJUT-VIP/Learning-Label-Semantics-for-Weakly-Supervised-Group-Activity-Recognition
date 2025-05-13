python test.py --dataset 'volleyball' --data_path './Dataset/' --num_frame 5 --num_total_frame 10 --num_activities 8 --motion --enc_layers 2 --nheads 2 --nheads_agg 2 --model_path './checkpoints/Volleyball_8_class.pth'
python test.py --dataset 'volleyball' --data_path '/root/autodl-tmp/Datasets/Volleyball_dataset/videos/' --num_frame 5 --num_total_frame 10 --num_activities 8 --motion --enc_layers 2 --nheads 2 --nheads_agg 2  --test_batch 6 --batch 6 --model_path '/root/autodl-tmp/Projects/tm/DF/result/[volleyball]_myb4/epoch24_90.65%.pth'

/root/autodl-tmp/Projects/tm/DF/result/selfa/epoch22_88.71%.pth
/root/autodl-tmp/Datasets/Collective_activity_dataset/ActivityDataset/