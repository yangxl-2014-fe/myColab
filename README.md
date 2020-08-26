
# Setup

```bash
conda create --name detectron2_env python=3.7.3
conda activate detectron2_env
conda info --envs
conda install pip
conda install ipython
pip install ninja yacs cython matplotlib tqdm opencv-python
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch


cd /disk4t0/DL-Home
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2


pip install google-auth==1.6.3
# pip install google-colab
pip install google-colab --use-feature=2020-resolver


pip uninstall google-colab
pip uninstall google-auth
pip install google-colab --use-feature=2020-resolver

pip install easydict
```
