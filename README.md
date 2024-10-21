# yolov8-autolabelimg
此工程沿用labelImg，只在autolabelimg中将yolov5修改为yolov8
若欲查看其他工具使用步骤，查看原工程：https://github.com/wufan-tb/AutoLabelImg

1. clone this repo：

   ```bash
   git clone https://github.com/alljoytech/yolov8-autolabelimg
   cd yolov8-autolabelimg
   ```

2. install requirments：

   ```bash
   conda create -n {your_env_name} python=3.7.6
   conda activate {your_env_name}
   pip install -r requirements.txt
   ```

3. compile source code：

   **Ubuntu User:**
   
   ```
   sudo apt-get install pyqt5-dev-tools
   make qt5py3
   ```
   
   **Windows User:**
   
   ```
   pyrcc5 -o libs/resources.py resources.qrc
   ```
   
4. prepare yolov5 weights file and move them to here: [official model zoo:[Yolov5](https://github.com/ultralytics/yolov5)]

   ```bash
   mv {your_model_weight.pt} pytorch_yolov5/weights/
   ```

5. open labelimg software

   ```
   python labelImg.py
   ```

## Set shortcut to open software[optional]

**Windows User:**

create a file:labelImg.bat, open it and type these text(D disk as an example)：

```bash
D:
cd D:{path to your labelImg folder}
start python labelImg.py
exit
```

double click labelImg.bat to open the software.

**Ubuntu User:**

open environment setting file: 

```bash
vim ~/.bashrc
```

add this command：

```bash
alias labelimg='cd {path to your labelImg folder} && python labelImg.py
```

source it：

```bash
source ~/.bashrc
```

typing 'labeling' in terminal to open the software.

## Citation

```
{   AutoLabelImg,
    author = {Wu Fan},
    year = {2020},
    url = {\url{https://https://github.com/wufan-tb/AutoLabelImg}}
}
```
