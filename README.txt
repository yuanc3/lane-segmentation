1、make_lmdb.py
该文件为制作lmdb数据集的代码
key：文件路径，value：文件的二进制数据内容，可采用opencv中的imdecode进行解码
注意lmdb的key和value均为bytes格式

另一种方式，将图片的numpy数组转为bytes，并将其shape、dtype一并存储进lmdb。但这种方式没有采用任何压缩算法，会使lmdb变得很大，因此不可取。

2、utils/image_process.py
代码中添加了一个新类LaneDatasetLMDB
其中在初始化阶段添加了lmdb的初始化代码，读取数据时从lmdb中读取文件二进制字节流，并用imdecode进行解码，需注意RGB图片和灰度图的不同。

3、train.py
将LaneDataset修改为LaneDatasetLMDB，需将lmdb文件路径传入。
注意在windows和mac中，采用多进程读取lmdb时会出错，而linux则可正常读取。