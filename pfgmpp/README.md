# PFGMPP

We perform a set of experiments of using multimodal datasets MNIST + CIFAR, CIFAR+OxfordIIITPet and so on on a different generative model governed by a seperate drift dynamics
which is [PFGM++: Unlocking the Potential of Physics-Inspired Generative Models](https://github.com/Newbeeer/pfgmpp/tree/main) the idea here is to model
particles as unit particles with the same charge and then use the dynamics of the resultant Electric Field $$\vec{E}$$ as the governing equation. 
This model has eseentially two free paramters for a given resolution (1) Architecture (ncsnpp | ddpm) (2) Augmented Dimension (D) where D governs the convergent distribution
as the authors show the $$ D \arrow \infty$$ is precisely a standard diffusion model. We chose 3 values of D (128,2048,$$\infty$$) as the authors have done
and perform the experiments for generation for each architecture for different datasets. This yields results that differ from the consistency/diffusion case for finite
D rather interestingly. Please refer to the mods branch for the modified code.

[Please access this folder for compressed datasets](https://drive.google.com/drive/folders/1KvJjlbAA3fENuJizVbHPiGakNsQ8ngYl?usp=drive_link). Use the compressed files as per the instructions in PFGMPP repo to train models.

## To Do:

- [ ] Add codes for rank calculation in bottleneck of U-NET
- [ ] Add codes for inversion/interpolation
- [ ] Add all zip files of datasets to drive

## Results:
We maintain the results in this link [Results for pfgmpp](https://docs.google.com/document/d/1JEIkwOn6OsS0MCICi-n6C309u5YczoCI5TPiX18X9Bo/edit#heading=h.836d5ejxumxo) for now and shall port to the repo later.

