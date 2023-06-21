# Diffusion Models

We observe a similar behavior as consistency models here perhaps due to the identical dynamics learnt. We use hugging face's diffusers implementation as our base while
trying out a couple of image diffusion models (DDPM,DDIM etc) and different sampling techniques eg PNDM.

For inversion currently we are using the forward process along with slerp for interpolation

## To-Do

- [ ] Add support for training of latent diffusion models eg D2C,DAE,LDM (Comp-vis)
- [ ] Add other inversion techniques eg DDIM Inversion 
