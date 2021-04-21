## HalentNet: Multimodal Trajectory Forecasting with Hallucinative Intents #
This repository contains the code of 
[HalentNet: Multimodal Trajectory Forecasting with Hallucinative Intents](https://openreview.net/forum?id=9GBZBPn0Jx&noteId=XdnOihdyKKU) 
for training and evaluation on nuScenes dataset.
This project is built upon this [repository](https://github.com/StanfordASL/Trajectron-plus-plus).




### Environment Setup ###

```
conda create -n halentnet python=3.6
conda activate halentnet
pip install -r requirements.txt
```

### Data Setup ###
Preprocessed nuScenes dataset can be found [here](https://drive.google.com/drive/folders/1tXCJUUTjCXALVvMcbuaUhdhM12Av7gE3?usp=sharing).
Download the files to `experiments/processed`. You can also download original nuScenes dataset and preprocess it by yourself following the instruction [here](https://github.com/StanfordASL/Trajectron-plus-plus).  

### Model Training ###
To train HalentNet on the nuScenes dataset, execute the following commands from within the `trajectron/` directory.

```
python train_halentnet.py --train_data_dict nuScenes_train_full.pkl  --eval_data_dict nuScenes_val_full.pkl --device cuda:0 --load_model ../experiments/nuScenes/models/int_ee_me --checkpoint 12 --preprocess_workers 4  --train_epochs 35  --log_tag halent   --log_dir /path/to/log/
```

### Model Evaluation ###
To evaluate the pretrained model, execute the following commands within the `experiments/nuScenes` directory.
A trained HalentNet can be found under the path `experiments/nuScenes/models/halentnet`. 
A pretrained base model (Trajectron++) can be found here `experiments/nuScenes/models/int_ee_me`.

```
python evaluate.py --model /path/to/model/ --checkpoint=checkpoint_to_evaluate --data ../processed/nuScenes_test_full.pkl --node_type VEHICLE --prediction_horizon 2 4 6 8 10 12
```