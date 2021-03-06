# slabTraining

# Synthetic Images of Longitudinal Cracks in Steel Slabs via Wasserstein Generative Adversarial Nets (GAN) used Towards Unsupervised Classification.
February 12, 2019

## Authors
* Diego Andrade - *Initial work* - [GANs](www.diegoandrade.org)
* Miguel Simiad - [ANT Automation](https://ant-automation.net/)


## Abstract
In metal forming procedures such as cold or hot rolling – which are processes where the metal stock is passed through multiple rolls to reduce the thickness and assuring uniformity – multiple modes of failure are expected, being longitudinal cracks the more prevalent. They can occur regardless of the care with the input slab but are easily detectable via human observation. Our goal is to train neural networks to classify these real defects. They are captured using digital cameras that become the raw data for any machine learning algorithm. In the industry, each continuous casters has its set of images due to a unique plant layout, illumination, camera angles, camera quality, sensitivity, environmental changes, occlusions, and so forth. All these possible contingencies make challenging to have a master neural network that captures all steel slab defects. The neural network learning rate becomes onerous, slow and pricey, being necessary to apply qualified human resources with in-depth knowledge of steel quality to train such net. As a consequence, we fall into a vicious cycle that is difficult to avoid, due to the lack of defect samples an adequately trained neural net is not produced, in turn, detection does not occur, thus skipping new potential real defects. Finally, the training rate slows or in the most unfavorable case, the neural net never finishes learning. With our proposed framework we lower the training time. We look at recent advances in machine learning to train neural networks via synthetic images using generative models. As far as we know, this is the first attempt of using a Wasserstein generative adversarial nets to create subsets of images for steel slab manufacturing. We use the “inception score” as a metric to validate our synthetic images. This approach is expected to save thousands of dollars and valuable time by capturing imperfections before they continue downstream in subsequent steps along the rolling process.

## Runnig the Code
To run the code just go to the root folder and run the following command using python.
```
python slabGANnGPU.py
```
### Real Image
![](/figures/256img.png?raw=true)

### Fake Image
![](/figures/longitudinalCrack159.jpg?raw=true)
