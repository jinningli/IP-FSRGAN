# FACESR
## Requirements
- python3
- torch 1.0.1
- torchvision 0.2.2
- Pretrained LightCNN model: download `LightCNN-29 v2` from https://github.com/AlfredXiangWu/LightCNN and place it in `recognition/pre-trained`

## training
```
python3 train.py --opt [path_to_training_config]
```
#### dataset
- An directory with all the high resolution images for training
- Should be specified in config file, such as `options/train/train_example.json`
```
...
,datasets": {
    "train": {
      "name": "VISHR"
      , "mode": "LRHR"
      , "resize": 0.9
      , "dataroot_HR": "/mnt/WXRG0235/jnli/datasets/VIS/VISHR/VISHR_train"
      //, "image_lists": ["/mnt/ficuszambia/jnli/facesr/datasets/zhengjian/train.list", "/mnt/WXRG0235/jnli/datasets/VIS/VISHR/VISHR_train.list"]
      , "subset_file": null
      , "use_shuffle": true
      , "n_workers": 8
      , "batch_size": 16
      , "HR_size": 128
      , "use_flip": true
      , "use_rot": true
      , "downsample": "numpy"
    }
...
```
- `subset_file` the list to HR images
- `name` the name of the training dataset
- `mode` LRHR: user can only provide HR, or provide both HR & LR
- `dataroot_HR` path to the directory
- `HR_size` size of patches random cropped for training
- `use_flip` `use_rot` flip or rotate
- `crop` to crop or just resize the whole face
- `resize` resize the init HR face (to find the best performance of LR size), 1 for not resize
- `downsample` downsample method for generating LR image. numpy | cubic | linear
- `image_lists` the path to all the HR images when dataroot_HR is not given.
#### path
```
"path": {
    "root": "/mnt/ficuszambia/jnli/facesr"
    // , "resume_state": "../experiments/debug_002_RRDB_ESRGAN_x4_DIV2K/training_state/16.state"
//    , "pretrain_model_G": "../experiments/pretrained_models/RRDB_PSNR_x4.pth"
  }
 ```
 - `root` the root path of this project for saving checkpoint (saving at root/Experiments)

 #### training settings
 ```
 , "train": {
    "lr_G": 1e-4
    , "weight_decay_G": 0
    , "beta1_G": 0.9
    , "lr_D": 1e-4
    , "weight_decay_D": 0
    , "beta1_D": 0.9
    , "lr_scheme": "MultiStepLR"
    , "lr_steps": [50000, 100000, 200000, 300000]
    , "lr_gamma": 0.5

    , "pixel_criterion": "l1"
    , "pixel_weight": 1e-2
    , "feature_criterion": "l1"
    , "feature_weight": 1
    , "gan_type": "vanilla"
    , "gan_weight": 5e-3
    , "recloss": {
      "name":"lightcnn",
      "path": "/mnt/ficuszambia/jnli/facesr/recognition/pre-trained/LightCNN_29Layers_V2_checkpoint.pth",
      "weight": 1
    }

    //for wgan-gp
    // , "D_update_ratio": 1
    // , "D_init_iters": 0
    // , "gp_weigth": 10

    , "manual_seed": 0
    , "niter": 1e6
    , "val_freq": 1000
  }
```
- `recloss-name` the name of recognition loss to be used. currently onlt `lightcnn` is supported. Leaving blank `''` for not using recognition loss
- `recloss-path` path to the pre-trained model
- `recloss-weight` weight of recloss, when set to `1`:
```
19-04-03 17:00:12.447 - INFO: <epoch:  0, iter:     200, lr:1.000e-04> l_g_pix: 4.4571e-03 l_g_fea: 2.6769e+00 l_g_rec: 6.0616e-01 l_g_gan: 1.3311e-01 l_d_real: 0.0000e+00 l_d_fake: 1.6391e-06 D_real: 2.3803e+01 D_fake: -2.8195e+00
...
19-04-03 20:50:43.945 - INFO: <epoch:  1, iter:  19,600, lr:1.000e-04> l_g_pix: 2.0350e-04 l_g_fea: 8.0707e-01 l_g_rec: 4.8295e-02 l_g_gan: 5.6367e-02 l_d_real: 7.4582e-04 l_d_fake: 7.7790e-05 D_real: -3.0652e+00 D_fake: -1.4338e+01

```
- `niter` how many batches to train?
- `val_freq` val image will be saved in `/valid`. Need manually deleted for each experiment
#### other settings
```
  "name": "lightcnn_VISHR_numpy_128crop" //  please remove "debug_" during training
  , "use_tb_logger": true
  , "model":"srragan"
  ,"scale": 4
  , "gpu_ids": [0,1,2,3,4,5,6]
```
- `name` checkpoints will be saved in `root/Experiments/name/models`. Attention: new experiment with the same name will overwrite the old one

## test
```
python3 test.py --opt [path_to_test_config]
```
#### testset
```
    "test 1": {
    "name": "FaceTest_Crop_1",
    "resize": 1,
    "mode": "LRHR",
     "downsample": "numpy",
      "dataroot_HR": "/mnt/WXRG0235/jnli/05_RegWorkspace/image/A40P/__backups_rgbfull__collected_crop"
  },
```
- `resize` resize the HR image before downsampling

### Reference
ESRGAN https://github.com/xinntao/BasicSR
