# Consistency Models

We try out multiple different implementations in a bid to understand the commonalities of bias in multimodal data. In particular we consider two and three dataset mixtures such as MNIST + CIFAR,
CIFAR + Oxford IIITPet, MNIST + OxfordIIITPet and so on. The idea being to observe behavior in terms of generation and interpolation and deviations from training sets. 
We observe a strong bias towards a single dataset in several cases which is reflected in the Effective Rank of the bottleneck of the underlying U-NET. The choice of this
layer is based on the observations in prior diffusion model works that show that it contains rich semantic information we hypothesize that as the CT-Model is just
an alternate way of learning the same PF-ODE with the same architecture this observation carries over as well. Further we observe a distinct difference in ranks between
diverse data sources such as training, interpolated and random data which continues more or less as the process evolves backwards in time. This difference ties into the 
observed bias as well. Above all the study raises questions on utilizing SDEs with unimodal convergent distribution for modelling unconditioned multimodal data. 


Added Implementations
- [x] Clone of Simo Link : 
- [x] Kinyugo Link :
- [x] Junhss Link :
- [ ] OpenAI (official) Link : 

Both Kinyugo and Junhss use Hugging Face's UNet2D Model from the diffusers library. Adding rank computation functions 

