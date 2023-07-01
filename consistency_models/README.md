# Consistency Models

We try out multiple different implementations in a bid to understand the commonalities of bias in multimodal data. In particular we consider two and three dataset mixtures such as MNIST + CIFAR,
CIFAR + Oxford IIITPet, MNIST + OxfordIIITPet and so on. The idea being to observe behavior in terms of generation and interpolation and deviations from training sets. 
We observe a strong bias towards a single dataset in several cases which is reflected in the Effective Rank of the bottleneck of the underlying U-NET. The choice of this
layer is based on the observations in prior diffusion model works that show that it contains rich semantic information we hypothesize that as the CT-Model is just
an alternate way of learning the same PF-ODE with the same architecture this observation carries over as well. Further we observe a distinct difference in ranks between
diverse data sources such as training, interpolated and random data which continues more or less as the process evolves backwards in time. This difference ties into the 
observed bias as well. Above all the study raises questions on utilizing SDEs with unimodal convergent distribution for modelling unconditioned multimodal data. 



## Implementation Details

Added Implementations
- [x] Clone of Simo Link : [link](https://github.com/cloneofsimo/consistency_models)
- [x] Kinyugo Link : [link](https://github.com/Kinyugo/consistency_models)
- [x] Junhss Link : [link](https://github.com/junhsss/consistency-models)
- [ ] OpenAI (official) Link : [link](https://github.com/openai/consistency_models)

Both Kinyugo and Junhss use Hugging Face's UNet2D Model from the diffusers library. We have added rank computation functions separately in utils.py. We will be adding inversion/interpolation functions soon for Junhss separately.We use the formula of computing the Grammian Matrix and then calculate the effective rank using that as per [this excellent work here](https://minyoungg.github.io/overparam/resources/overparam-v3.pdf). Please use this after features have been extracted using given forward hook implementations. For clone of simo feature extraction has been directly added to the core U-NET class. Please only refer to the modifications branch in the attached forked repository of Clone of simo's implementation. This has all the changes. This is **not** in sync with the master branch as the changes are not official by any means.


Currently OpenAI's implementation and some JAX based implementations are not yet supported. OpenAI has a relatively complicated codebase that will take some time to port and experiment with. CD based training support has also not been added as this requires a separate diffusion model training step. 

## Results 

Attaching a link to a google doc here which is [temporary until results are ported to this repository](https://docs.google.com/document/d/1JEIkwOn6OsS0MCICi-n6C309u5YczoCI5TPiX18X9Bo/edit) Please do **not** modify without permission. Once all results are updated in the doc they will be ported to this repo. 

### Update 
We conjecture that intrinsic dimension of data is a key factor contributing to this set of observations. To this end we construct an artificial dataset [available at this link](https://drive.google.com/drive/folders/1n1aOUuNAq6sL6sGHC5dc38cGdZO32fdK) for public experimentation. Following this excellent work from Pope et.al we construct our dataset by generating images from a BigGAN trained on imagenet for a specific class, Basenji with different amounts of latents masked. This allows us to create datasets of varying intrinsic dimensionality (in this case 16,32,64 and 128) but with the same visual content. We observe the effect of dimensionality, number of samples and mixtures of varying dimensionality to asceratin this.  
