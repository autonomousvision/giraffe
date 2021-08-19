# GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields
## [Project Page](https://m-niemeyer.github.io/project-pages/giraffe/index.html) | [Paper](http://www.cvlibs.net/publications/Niemeyer2021CVPR.pdf) | [Supplementary](http://www.cvlibs.net/publications/Niemeyer2021CVPR_supplementary.pdf) | [Video](http://www.youtube.com/watch?v=fIaDXC-qRSg&vq=hd1080&autoplay=1) | [Slides](https://m-niemeyer.github.io/slides/talks/giraffe/index.html) | [Blog](https://autonomousvision.github.io/giraffe/) | [Talk](https://www.youtube.com/watch?v=scnXyCSMJF4)
![Add Clevr](gfx/add_clevr6.gif)
![Tranlation Horizontal Cars](gfx/tr_d_cars.gif)
![Interpolate Shape Faces](gfx/rotation_celebahq.gif)

If you find our code or paper useful, please cite as

    @inproceedings{GIRAFFE,
        title = {GIRAFFE: Representing Scenes as Compositional Generative Neural Feature Fields},
        author = {Niemeyer, Michael and Geiger, Andreas},
        booktitle = {Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
        year = {2021}
    }

## TL; DR - Quick Start

![Rotating Cars](gfx/rotation_cars.gif)
![Tranlation Horizontal Cars](gfx/tr_h_cars.gif)
![Tranlation Horizontal Cars](gfx/tr_d_cars.gif)

First you have to make sure that you have all dependencies in place. The simplest way to do so, is to use [anaconda](https://www.anaconda.com/).

You can create an anaconda environment called `giraffe` using
```
conda env create -f environment.yml
conda activate giraffe
```

You can now test our code on the provided pre-trained models.
For example, simply run
```
python render.py configs/256res/cars_256_pretrained.yaml
```
This script should create a model output folder `out/cars256_pretrained`.
The animations are then saved to the respective subfolders in `out/cars256_pretrained/rendering`.

## Usage

### Datasets

To train a model from scratch or to use our ground truth activations for evaluation, you have to download the respective dataset.

For this, please run
```
bash scripts/download_dataset.sh
```
and following the instructions. This script should download and unpack the data automatically into the `data/` folder.


### Controllable Image Synthesis

To render images of a trained model, run
```
python render.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the correct config file.
The easiest way is to use a pre-trained model.
You can do this by using one of the config files which are indicated with `*_pretrained.yaml`. 

For example, for our model trained on Cars at 256x256 pixels, run
```
python render.py configs/256res/cars_256_pretrained.yaml
```
or for celebA-HQ at 256x256 pixels, run
```
python render.py configs/256res/celebahq_256_pretrained.yaml
```
Our script will automatically download the model checkpoints and render images.
You can find the outputs in the `out/*_pretrained` folders.

Please note that the config files  `*_pretrained.yaml` are only for evaluation or rendering, not for training new models: when these configs are used for training, the model will be trained from scratch, but during inference our code will still use the pre-trained model.

### FID Evaluation
For evaluation of the models, we provide the script `eval.py`. You can run it using
```
python eval.py CONFIG.yaml
```
The script generates 20000 images and calculates the FID score.

Note: For some experiments, the numbers in the paper might slightly differ because we used the evaluation protocol from [GRAF](https://github.com/autonomousvision/graf) to fairly compare against the methods reported in GRAF.

### Training
Finally, to train a new network from scratch, run
```
python train.py CONFIG.yaml
```
where you replace `CONFIG.yaml` with the name of the configuration file you want to use.

You can monitor on <http://localhost:6006> the training process using [tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard):
```
cd OUTPUT_DIR
tensorboard --logdir ./logs
```
where you replace `OUTPUT_DIR` with the respective output directory. For available training options, please take a look at `configs/default.yaml`.

## 2D-GAN Baseline

For convinience, we have implemented a 2D-GAN baseline which closely follows this [GAN_stability repo](https://github.com/LMescheder/GAN_stability). For example, you can train a 2D-GAN on CompCars at 64x64 pixels similar to our GIRAFFE method by running
```
python train.py configs/64res/cars_64_2dgan.yaml
```

## Using Your Own Dataset

If you want to train a model on a new dataset, you first need to generate ground truth activations for the intermediate or final FID calculations.
For this, you can use the script in `scripts/calc_fid/precalc_fid.py`.
For example, if you want to generate an FID file for the comprehensive cars dataset at 64x64 pixels, you need to run
```
python scripts/precalc_fid.py  "data/comprehensive_cars/images/*.jpg" --regex True --gpu 0 --out-file "data/comprehensive_cars/fid_files/comprehensiveCars_64.npz" --img-size 64
```
or for LSUN churches, you need to run
```
python scripts/precalc_fid.py path/to/LSUN --class-name scene_categories/church_outdoor_train_lmdb --lsun True --gpu 0 --out-file data/church/fid_files/church_64.npz --img-size 64
```

Note: We apply the same transformations to the ground truth images for this FID calculation as we do during training. If you want to use your own dataset, you need to adjust the image transformations in the script accordingly. Further, you might need to adjust the object-level and camera transformations to your dataset. 

## Evaluating Generated Images

We provide the script `eval_files.py` for evaluating the FID score of your own generated images.
For example, if you would like to evaluate your images on CompCars at 64x64 pixels, save them to an npy file and run
```
python eval_files.py --input-file "path/to/your/images.npy" --gt-file "data/comprehensive_cars/fid_files/comprehensiveCars_64.npz"
```

# Futher Information

## More Work on Implicit Representations
If you like the GIRAFFE project, please check out related works on neural representions from our group:
- [Schwarz et. al. - GRAF: Generative Radiance Fields for 3D-Aware Image Synthesis (NeurIPS'20)](https://github.com/autonomousvision/graf)
- [Niemeyer et. al. - DVR: Learning Implicit 3D Representations without 3D Supervision (CVPR'20)](https://github.com/autonomousvision/differentiable_volumetric_rendering)
- [Oechsle et. al. - Learning Implicit Surface Light Fields (3DV'20)](https://arxiv.org/abs/2003.12406)
- [Peng et. al. - Convolutional Occupancy Networks (ECCV'20)](https://arxiv.org/abs/2003.04618)
- [Niemeyer et. al. - Occupancy Flow: 4D Reconstruction by Learning Particle Dynamics (ICCV'19)](https://avg.is.tuebingen.mpg.de/publications/niemeyer2019iccv)
- [Oechsle et. al. - Texture Fields: Learning Texture Representations in Function Space (ICCV'19)](https://avg.is.tuebingen.mpg.de/publications/oechsle2019iccv)
- [Mescheder et. al. - Occupancy Networks: Learning 3D Reconstruction in Function Space (CVPR'19)](https://avg.is.tuebingen.mpg.de/publications/occupancy-networks)
