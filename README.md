# Deep Exemplar-based Colorization

This is the implementation of paper [**Deep Exemplar-based Colorization**](https://arxiv.org/abs/1807.06587) by Mingming He*, [Dongdong Chen*](http://www.dongdongchen.bid/),
[Jing Liao](https://liaojing.github.io/html/index.html), [Pedro V. Sander](http://www.cse.ust.hk/~psander/) and 
[Lu Yuan](http://www.lyuan.org/) in ACM Transactions on Graphics (SIGGRAPH 2018) (*indicates equal contribution).


## Introduction

**Deep Exemplar-based Colorization** is the ﬁrst deep learning approach for exemplar-based local colorization. 
Given a reference color image, our convolutional neural network directly maps a grayscale image to an output colorized image.

![image](https://github.com/msracver/Deep-Exemplar-based-Colorization/blob/master/demo/data/representative.jpg)

The proposed network consists of two sub-networks, **Similarity Sub-net** which computes the semantic similarities between 
the reference and the target, and **Colorization Sub-net** which selects, propagates and predicts the chrominances channels of the target.

The input includes a grayscale target image, a color reference image and bidirectional mapping functions. We use [*Deep Image Analogy*](https://github.com/msracver/Deep-Image-Analogy) as default to generate bidirectional mapping functions. It is applicable to replace with other dense correspondence estimation algorithms.

The code of the part **Color Reference Recommendation** is now released. Please refere to [Gray-Image-Retrieval](https://github.com/hmmlillian/Gray-Image-Retrieval) for more details.

For more results, please refer to our [Supplementary](http://www.dongdongchen.bid/supp/deep_exam_colorization/index.html).

The code of the part **Color Reference Recommendation** is now released. Please refere to [Gray-Image-Retrieval](https://github.com/hmmlillian/Gray-Image-Retrieval) for more details.

(**Update**) Many thanks to [jqueguiner](https://github.com/jqueguiner) for adding support for Docker on Linux.


## License

© Microsoft, 2017. Licensed under a MIT license.


## Linux Support (By [jqueguiner](https://github.com/jqueguiner)(Colorization Subnet), [ncianeo](https://github.com/ncianeo) (Similarity Combo and Deep Image Analogy))

### Demo for Linux / Docker

#### Building the docker
```
docker build -t deep-colorization -f Dockerfile .
```

#### Before running
If you want to run the provided demo:

This section requires docker version >= 19.03.
Then, setup [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker) to use cuda support.

#### Running the docker for the demo
```
docker run -it --ipc=host --gpus=all deep-colorization
```

#### Running the demo
Once in the Docker
```
root@84ccb98c1b2e:/src/app# ls
colorization_subnet  demo  requirements.txt  similarity_subnet
root@84ccb98c1b2e:/src/app# cd demo/
root@84ccb98c1b2e:/src/app/demo# ls
data  example models  run.sh
root@84ccb98c1b2e:/src/app/demo# ./run.sh
```

#### Inputs
Inputs look like:
```
root@3a808ffe15a4:/src/app/demo/example/input# ls
in1.jpg  in2.JPEG  ref1.jpg  ref2.JPEG
```
with in*.jpg being the original images to colorize and ref*.jpg the colorized image to transfer from.

#### Outputs
Outputs will be place under the /src/app/demo/example/res folder

```
root@3a808ffe15a4:/src/app/demo/example/res# ls
in1_ref1.png  in2_ref2.png
```

#### Running the demo on your local images
If you want to run on your custom local images,
```
docker run -it --ipc=host -v /your/local/path/to/images:/src/app/custom_example deep-colorization
```

Once in the docker
```
/src/app/demo/run.sh /src/app/custom_example
```

## Citation
If you find **Deep Exemplar-based Colorization** helpful for your research, please consider citing:
```
@article{he2018deep,
  title={Deep exemplar-based colorization},
  author={He, Mingming and Chen, Dongdong and Liao, Jing and Sander, Pedro V and Yuan, Lu},
  journal={ACM Transactions on Graphics (TOG)},
  volume={37},
  number={4},
  pages={47},
  year={2018},
  publisher={ACM}
}
```
