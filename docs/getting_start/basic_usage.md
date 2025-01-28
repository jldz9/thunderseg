Thunderseg offers both CLI entry point and Python API for production and research, users who are familiar with Python could build their own functions and modules on top of Thunderseg.
## CLI usage

If Thunderseg is properly installed, you should be able to run command `thunderseg` in CLI, try:

```bash
thunderseg --help 
```

*[CLI]: Command-Line Interface
*[API]:Application Programming Interface

In order to run this tutorial, you can download example dataset to your home directory by using: 
```bash
thunderseg -e 
``` 
After download completed, a `thunderseg_example_data` folder should exist in your home directory with structure: 

```bash
thunderseg_example_data
├── predict  #Place to put your prediction raster
|   └── Drake20220819_MS.tif
├── shp
│   ├── train_shp # Place to put your training shape file 
│   │   ├── shp_20220928.cpg
│   │   ├── shp_20220928.dbf
│   │   ├── shp_20220928.prj
│   │   ├── shp_20220928.shp
│   │   └── shp_20220928.shx
│   ├── valid_shp # Place to put your validation shape file 
│   │   ├── valid.cpg
│   │   ├── valid.dbf
│   │   ├── valid.prj
│   │   ├── valid.shp
│   │   └── valid.shx
│   └── test_shp 
└── train # Place to put your training raster
    └── Drake20220928_MS.tif
```
This is the default directory structure to prepare for Thunderseg program. 
Each folder inside workdir directory is explained below: 

| Name | Note| necessity| 
|-----|----|---|
|`predict`| Place to put your raster for predict process, can be multiple raster|optional|
|`shp/train_shp`| Place to put your Shapefile for training purpose, can be multiple Shapefile| required|
|`shp/valid_shp`| Place to put your Shapefile for validate purpose, can be multiple Shapefile| if empty will split train Shapefile for valid process|
|`shp/test_shp` | Place to put your Shapefile for test purpose, can be multiple Shapefile|if empty will split valid Shapefile for test process|
|`train`|Place to put your raster for train process, can be multiple raster | required|

### > Step 1: Copy and Setup Config
Thunderseg CLI entry point uses configure file through the entire process. 
Run below command to get default configuration file copied to `thunderseg_example_data` folder.
```bash
thunderseg -g ~/thunderseg_example_data
```

You should find a `config.toml` file under `thunderseg_example_data` folder looks like below: 

??? example
    ```toml
    # This is the config file for thunderseg program. Program use metric system
    [IO]
    WORKDIR = "/" # Your Path to the working directory
    TRAIN_RASTER_DIR = "/" # Your Path to the training raster directory
    TRAIN_SHP_DIR = "/" # Your Path to the training Shapefile directory
    VALID_SHP_DIR = "/" # Your Path to the validation Shapefile directory
    PREDICT_RASTER_DIR = "/" # Your Path to the prediction raster directory

    [PREPROCESS]
    TILE_SIZE = 100 # meter
    BUFFER_SIZE = 10 # meter
    DEBUG = false
    MODE = "BGR"

    [PREPROCESS.RESAMPLE]
    ENABLE = false
    RESOLUTION = 0.1 # meter

    [PREPROCESS.TRANSFORM]
    RANDOM_CROP_HEIGHT = 512
    RANDOM_CROP_WIDTH = 512

    [PREPROCESS.COCO_INFO]
    NAME = "testdataset"
    VERSION = "1.0"
    DESCRIPTION = "This is a test dataset"
    CONTRIBUTOR = "unknown"
    URL = "unknown"
    # You can add as much descriptions as you want under this section.
    # e.g.
    # time = '2019-08-24'
    # email = 'myemail@email.com'

    [TRAIN]
    MODEL = "maskrcnn_rgb" # other model option in the furture
    MAX_EPOCHS = 100
    NUM_CLASSES = 2
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 5
    NUM_WORKERS = 15
    ```
You need to change PATH in IO section in the config file to corresponding paths in `thunderseg_example_data` directory: 

??? example
    <span style= "color: red;"> Make sure you use your own path instead of copy the exact same path provided below, or you will get file not found error later</span>. You can use `pwd` to find path of your current work directory.
    ```toml
    # This is the config file for thunderseg program. Program use metric system
    [IO]
    WORKDIR = "/home/jldz9/thunderseg_example_data/workdir" # Your Path to the working directory
    TRAIN_RASTER_DIR = "/home/jldz9/thunderseg_example_data/train" # Your Path to the training raster directory
    TRAIN_SHP_DIR = "/home/jldz9/thunderseg_example_data/shp/train_shp" # Your Path to the training Shapefile directory
    VALID_SHP_DIR = "/home/jldz9/thunderseg_example_data/shp/valid_shp" # Your Path to the validation Shapefile directory
    PREDICT_RASTER_DIR = "/home/jldz9/thunderseg_example_data/predict" # Your Path to the prediction raster directory

    [PREPROCESS]
    TILE_SIZE = 100 # meter
    BUFFER_SIZE = 10 # meter
    DEBUG = false
    MODE = "BGR"

    [PREPROCESS.RESAMPLE]
    ENABLE = false
    RESOLUTION = 0.1 # meter

    [PREPROCESS.TRANSFORM]
    RANDOM_CROP_HEIGHT = 512
    RANDOM_CROP_WIDTH = 512

    [PREPROCESS.COCO_INFO]
    NAME = "testdataset"
    VERSION = "1.0"
    DESCRIPTION = "This is a test dataset"
    CONTRIBUTOR = "unknown"
    URL = "unknown"
    # You can add as much descriptions as you want under this section.
    # e.g.
    # time = '2019-08-24'
    # email = 'myemail@email.com'

    [TRAIN]
    MODEL = "maskrcnn_rgb" # other model option in the furture
    MAX_EPOCHS = 100
    NUM_CLASSES = 2
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 5
    NUM_WORKERS = 15
    ```
### > Step 2: Run Preprocess

Preprocess module tiles input raster into same size tiles and then convert to a COCO dataset.
*[COCO]: Common Objects in Context 

Run code below to start preprocess: 
```bash
thunderseg preprocess -c ~/thunderseg_example_data/config.toml
```
A `workdir` will be generated under `thunderseg_example_data`, the directory structure will look like below:
??? example 
    ```bash 
    ├── config.toml
    ├── predict
    ├── shp
    │   ├── train_shp
    │   │   ├── shp_20220928.cpg
    │   │   ├── shp_20220928.dbf
    │   │   ├── shp_20220928.prj
    │   │   ├── shp_20220928.qix
    │   │   ├── shp_20220928.qmd
    │   │   ├── shp_20220928.shp
    │   │   └── shp_20220928.shx
    │   └── valid_shp
    │       ├── valid.cpg
    │       ├── valid.dbf
    │       ├── valid.prj
    │       ├── valid.shp
    │       └── valid.shx
    ├── train
    │   └── Drake20220928_MS.tif
    └── workdir
        ├── datasets
        │   ├── annotations
        │   │   ├── train_coco.json
        │   │   └── valid_coco.json
        │   ├── predict
        │   ├── train
        │   │   ├── Drake20220928_MS_row0_col0.png
        │   │   ├── Drake20220928_MS_row0_col0.tif
        │   │   ├── Drake20220928_MS_row0_col11151.png
        │   │   ├── Drake20220928_MS_row0_col11151.tif
        │   │   ├── Drake20220928_MS_row0_col1239.png
        │   │   ├── Drake20220928_MS_row0_col1239.tif
        │   │   ├── .......
        │   │   └── shp
        │   │       ├── train_shp.cpg
        │   │       ├── train_shp.dbf
        │   │       ├── train_shp.prj
        │   │       ├── train_shp.shp
        │   │       └── train_shp.shx
        │   └── val
        │       └── shp
        |           ├── valid_shp.cpg
        │           ├── valid_shp.dbf
        │           ├── valid_shp.prj
        │           ├── valid_shp.shp
        │           └── valid_shp.shx
        ├── results
        └── temp
            ├── Drake20220928_MS_coco.json
            └── Drake20220928_MS_coco_valid.json
    ```

Each folder under workdir directory is explained below: 

| Name| Note | 
|---------|------|
|`dataset` | Storage all output data during the process|
|`dataset/annotation` | Stitched COCO dataset based on user provide Shapefile|
|`dataset/predict`| Place to storage predict tiles during predict stage| 
|`dataset/train`| Place to storage train tiles during preprocess and train stage|
|`dataset/train/shp`| Stitched training Shapefile|
|`dataset/val/shp`| stitched validation Shapefile|
|`results`| Place to storage all final products and logs|
|`temp` | Place to storage temporary files for debug uses| 

`*png` files generated during processing is used to preview the tiling result. 
### > Step 3: Run Train

Train process use model script as plugins, you can prepare your own model script (:warning: <span style= "color: orange;"> Not available in preview version yet </span>)

You can call built in model script by using corresponding model name in the config file under TRAIN section e.g.: 
```toml
[TRAIN]
MODEL = "maskrcnn_rgb" 
```
OR you can create your own model script and point MODEL to absolute path of your model script, You can check more info about how to [prepare your own model script]()
```toml
[TRAIN]
MODEL = "/home/jldz9/mymodels/foo.py" 
```

Currently, Thunderseg has one built in model scripts for instance segmentation

| Model | description  |
|---|---|
| [maskrcnn_rgb]()| A classic instance segmentation model, use maskrcnn_resnet50_fpn_v2 from torchvision | 

Once select the model script in the config file, run: 

```bash
thunderseg train -c ~/thunderseg_example_data/config.toml
```
The training process will start, and you should be able to check your training process via Tensorboard at `http://localhost:6006/` in browser.

Once training process is finished, the model state will be saved as a checkpoint file (`.ckpt`) inside the log folder, you can check pytorch-lightning documentation for [more description of checkpoint file](https://lightning.ai/docs/pytorch/stable/common/checkpointing.html#checkpointing). The output directory structure should look like below: 

??? example 
    ```bash
    ├── results
    │   └── logs
    │       ├── version_0
    │       │   ├── checkpoints
    │       │   │   └── epoch=0-step=22.ckpt
    │       │   ├── events.out.tfevents.1738012420.DESKTOP-RHDONR9.11097.0
    │       │   └── hparams.yaml
    │       └── version_1
    │           ├── checkpoints
    │           │   └── epoch=0-step=22.ckpt
    │           ├── events.out.tfevents.1738012523.DESKTOP-RHDONR9.16628.0
    │           └── hparams.yaml
    ```

|Name | Note|
|---|---|
|`logs` | Place to storage all training logs|
|`logs/version_0` or `logs/version_*`| Each version represent a training model, if you run the training multiple times there will be multiple versions of the model|
|`logs/version_0/checkpoints`| The place to save checkpoint file, this is your trained model|
|`events.out.tfevents*`| The tensorboard log file to track your logs during training|
|` hparams.yaml`| Storage all hyperparameters used in training process|

### > Step 4: Run Predict and Post-processing 

Once you have your model trained, run code below to predict and post-process images: 

```bash 
thunderseg predict -c ~/thunderseg_example_data/config.toml --ckpt ~/thunderseg_example_data/workdir/results/logs/version_1/checkpoints/epoch=0-step=22.ckpt
```
``--ckpt`` option is used to load your checkpoint file form training process. 
If ``--ckpt`` is not provided, Thunderseg will attempt to find the latest check-point file storage under `result` directory.

After the process, you should be able to find two new files generated under results folder as a `*gpkg` file, which is a geo-package file that used to load into GIS softwares.

*[GIS]: Geographic Information System

```bash
result
├── Drake20220819_MS_bounding_box.gpkg
├── Drake20220819_MS_polygon.gpkg
```

|folder | Note|
|---|---|
|`*_bounding_box.gpkg`| The bounding box prediction|
|`*_polygon.gpkg`| The mask prediction | 

That's all, you've completed the whole training loop! 

## API usage

The API document is under development




