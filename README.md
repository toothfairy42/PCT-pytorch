## PCT
This is a Pytorch implementation of PCT: Point Cloud Transformer.

Paper link: https://arxiv.org/pdf/2012.09688.pdf

- only classification for now
- runs on cpu (takes forever)

### Requirements
python >= 3.7

pytorch >= 1.6

h5py

scikit-learn

### Example training and testing
```shell script
# train for classification
python3 main.py --exp_name=train --num_points=1024 --use_sgd=True --batch_size 32 --epochs 250 --lr 0.0001

# test for classification
python3 main.py --exp_name=test --num_points=1024 --use_sgd=True --eval=True --model_path=checkpoints/best/models/model.t7 --test_batch_size 8

# train for segmentation on airplanes
python3 main_seg.py --exp_name train --class_choice airplane  --num_points 1024 --use_sgd=True --batch_size 32 --epochs 250 --lr 0.0001

# test for segmentation on airplanes
python3 main_seg.py --exp_name test --class_choice airplane --num_points 1024 --eval=True --model_path=checkpoints/best/models/model.t7 --test_batch_size 8

```

