# HPC Instructions

## Discovery cluster information 
We will use CARC’s general-use HPC cluster "Discovery".

There are 2 ways to access this cluster: 
- SSH to the Discovery Cluster: You need to SSH from your terminal to the remote cluster. Please read the following [guide](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/getting-started-discovery)
- OnDemand: This provides a graphical, browser-based access. Please read the following [guide](https://www.carc.usc.edu/user-guides/carc-ondemand/ondemand-overview)

## Connect to the USC network or Research VPN. [guide](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/getting-started-discovery#connecting-to-the-usc-network-or-research-vpn)
You will have to either be connected to the `USC Secure Wireless` network or use the [USC VPN](https://www.carc.usc.edu/user-guides/quick-start-guides/anyconnect-vpn-setup) to connect to the cluster.

## Log in to Discovery node and access the terminal:
There are 2 ways:
- SSH to the Discovery Cluster: from a terminal on your own machine, run the following (replace `<netid>` with your USC NetID; you will be asked for your password and Duo). Refer to the [instructions](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/getting-started-discovery#logging-in) for OS-specific details.
    ```shell
    $ ssh <netid>@discovery.usc.edu
    ```
- OnDemand: 
    1. Use this [link](https://ondemand.carc.usc.edu/) to login using USC NetID.
    2. Click on Home Directory
    3. Open the terminal in the Home Directory. 
    
In both ways, you should get `[username@discovery* ~]` prompt in the terminal.

**Note: It is important to be familiar with how to use the shell for each case, and also how to transfer files.**

## Conventions:
In the rest of the document, we will assume the following convention:

- In code blocks, lines that start with `$` are shell commands. Copy the text after the `$`.
- In code blocks, lines that start with `#` are either outputs or comments.
- Output code in `<...>` are placeholders. In output lines that start with `#` just refers to text we don't necessarily care about.

## Initial Setup

Before you can run the experiments, you will have to setup the environment for the HPC.

### Python/Anaconda setup

Mini-Project 2 needs **Python 3.10** (Mini-Project 1 used Python 3.8), so you will create a **new, separate** conda environment called `carla-mp2`. Do not reuse or delete the `carla` environment from Mini-Project 1.

If you set up conda on Discovery for Mini-Project 1, the `conda` command is already available after you log in. Check with:

```shell
$ source ~/.bashrc
$ conda --version
```

If you get `conda: command not found` (for example, because you did Mini-Project 1 on your own machine), run the following once and then continue:

```shell
$ module purge
$ module load conda
$ conda init bash
$ conda config --set auto_activate_base false # This will prevent the base environment from being loaded
$ source ~/.bashrc
```

Now create the environment. We give its full path so that it lands in the same place for everyone:

```shell
$ conda create -p ~/.conda/envs/carla-mp2 python=3.10 -c conda-forge -y
```

To use this environment, run:

```shell
$ conda activate ~/.conda/envs/carla-mp2
```

**Note:** Always activate the environment by its full path as shown above, not by name. The prompt prefix is not a reliable indicator of which environment is active, so verify with the following commands:

```shell
$ python --version
# Python 3.10.<*>
$ which python
# ~/.conda/envs/carla-mp2/bin/python
```

### Code transfer

Get the template for the project and unpack it in your home directory:

```shell
$ cd ~
$ cp /project2/jdeshmuk_786/csci513-miniproject2.zip .
$ unzip csci513-miniproject2.zip
```

This will create a `csci513-miniproject2` directory in your home directory.

The pretrained YOLOv3 object detector is too large to ship inside the archive. Copy it into the project:

```shell
$ cp /project2/jdeshmuk_786/yolo.h5 ~/csci513-miniproject2/mp2_distance_predictor/yolo.h5
```

Finally, copy your controller from Mini-Project 1 into `~/csci513-miniproject2/mp2_controller/controller.py`. The only change required is the import line at the top, which must read `from mp2_simulator.simulator import Observation, Mode`.

### Acquire a compute node

Run the following command to acquire a compute "node" with a GPU:

```shell
$ salloc --time=2:00:00 --cpus-per-task=8 --mem=32GB --account=jdeshmuk_786 --partition=gpu --gres=gpu:1
# salloc: Granted job allocation <job id>
# salloc: Waiting for resource configuration
# salloc: Nodes <node-name> are ready for job
```

Run the command `nvidia-smi` to ensure the GPU drivers were loaded properly:

```shell
$ nvidia-smi
```

Load necessary dependencies

```shell
$ module load apptainer
$ module load cuda
$ module load cudnn
$ export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
```

### Carla setup

Activate carla environment

```shell
$ source ~/.bashrc
$ conda activate ~/.conda/envs/carla-mp2
$ which python # must print ~/.conda/envs/carla-mp2/bin/python
```

Install dependencies (only needed the first time; this downloads TensorFlow and takes a few minutes)

```shell
$ cd ~/csci513-miniproject2
$ python3 -m pip install -U -e .
```

This time we will run the CARLA server inside the singularity container while staying outside of it:

```shell
$ singularity exec --nv /project2/jdeshmuk_786/carla-0.9.15_4.11.sif bash /home/carla/CarlaUE4.sh -nosound -vulkan -RenderOffScreen & # The & is required
# 4.26.2-0+++UE4+Release-4.26 522 0
# Disabling core dumps.

# Hit ENTER here and the prompt will reappear. Wait about 30 seconds for CARLA to finish loading.
```

Now you can run the project scripts as described in the README: train your two models, then run the simulator and the evaluator from `~/csci513-miniproject2`.

## Coming back later

On later connections to HPC you can skip creating the conda environment and installing the python dependencies. Every new session needs only the following:

```shell
$ salloc --time=2:00:00 --cpus-per-task=8 --mem=32GB --account=jdeshmuk_786 --partition=gpu --gres=gpu:1
$ module load apptainer
$ module load cuda
$ module load cudnn
$ export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
$ source ~/.bashrc
$ conda activate ~/.conda/envs/carla-mp2
$ singularity exec --nv /project2/jdeshmuk_786/carla-0.9.15_4.11.sif bash /home/carla/CarlaUE4.sh -nosound -vulkan -RenderOffScreen &
$ cd ~/csci513-miniproject2
```

Training the two small models does not need CARLA or a GPU allocation to be running, but it does need the `carla-mp2` environment to be active.

Your home directory is the same filesystem on the login node and on the GPU node, and the project is installed in editable mode, so any file you save takes effect the next time you run a script. As in Mini-Project 1, the recommended workflow is one terminal on the GPU node for running things and a second login to Discovery (another `ssh` session, the OnDemand editor, or VS Code *Remote - SSH*) for editing.

**Note:** the GPU allocation from `salloc` expires after the requested time (2 hours in the examples above). Your files are not affected, but you will need to run `salloc` again and relaunch CARLA to continue simulating.

## Downloading your files for submission

You submit five files on Brightspace (see "What to submit" in the README). To copy them from Discovery to your own machine, run the following **from a terminal on your own machine**, not on Discovery (replace `<netid>` with your USC NetID):

```shell
$ scp <netid>@discovery.usc.edu:csci513-miniproject2/mp2_controller/controller.py .
$ scp <netid>@discovery.usc.edu:csci513-miniproject2/mp2_distance_predictor/in_lane_model.h5 .
$ scp <netid>@discovery.usc.edu:csci513-miniproject2/mp2_distance_predictor/distance_model.h5 .
$ scp <netid>@discovery.usc.edu:csci513-miniproject2/mp2_distance_predictor/train_in_lane_model.py .
$ scp <netid>@discovery.usc.edu:csci513-miniproject2/mp2_distance_predictor/train_distance_model.py .
```

Alternatively, use the file browser in OnDemand (Files, then Home Directory) and download each file.

## Keep your work private

Keep your project in your **home directory**, as these instructions do. Do not copy your code or models into `/project2/jdeshmuk_786/`: that directory is shared with the whole class, so anything you put there can be read by other students.
