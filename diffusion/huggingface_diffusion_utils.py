import torch 
import numpy as np
import torch.nn.functional as F





def forward_process(noise_scheduler,clean_images,t):

    """

    For use with a given noise scheduler and a batch of clean images, this function returns the corresponding noisy images at a time t.
    
    
    """
    timesteps = (torch.ones((clean_images.shape[0]))*t).long().to(clean_images.device)
    noise = torch.randn_like(clean_images).to(clean_images.device)
    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)


    return noisy_images

def cos(a, b):
    a = a.view(-1)
    b = b.view(-1)
    a = F.normalize(a, dim=0)
    b = F.normalize(b, dim=0)
    return (a * b).sum()



def get_interpolated_images_slerp(noise_scheduler,clean_images_1,clean_images_2,t,n_steps):

    """
    Send tensors of same shape to this function. If length of shapes are 3 returns a tensor of size n_steps,3,H,W else if both tensors have a batch size of B
    returns a tensor of size B*n_steps,3,H,W. Interpolation is linear
    
    
    """
    noisy_images_1 = forward_process(noise_scheduler,clean_images_1,t)
    noisy_images_2 = forward_process(noise_scheduler,clean_images_2,t)

    d1 = len(clean_images_1.shape)
    d2 = len(clean_images_2.shape)

    if d1 == 3 and d2 == 3:
        noisy_images_1 = noisy_images_1.view(1,3,clean_images_1.shape[1],clean_images_1.shape[2])
        noisy_images_2 = noisy_images_2.view(1,3,clean_images_1.shape[1],clean_images_1.shape[2])
    elif d1 == 4 and d2 == 4:
        noisy_images_1 = noisy_images_1
        noisy_images_2 = noisy_images_2
    else:
        raise NotImplementedError


    alpha = torch.tensor(np.linspace(0, 1, n_steps, dtype=np.float32))



    theta = torch.arccos(cos(noisy_images_1, noisy_images_2))
    x_shape = noisy_images_1.shape
    intp_x = (torch.sin((1 - alpha[:, None]) * theta) * noisy_images_1.flatten(0, 2)[None].cpu() + torch.sin(alpha[:, None] * theta) * noisy_images_2.flatten(0, 2)[None].cpu()) / torch.sin(theta)
    intp_x = intp_x.view(-1, *x_shape).to(noisy_images_1.device)
    if d1 == 3 and d2 == 3:
        intp_x = intp_x.view(n_steps,3,clean_images_1.shape[1],clean_images_1.shape[2])
    elif d1 == 4 and d2 == 4:
        intp_x = intp_x.view(-1,3,clean_images_1.shape[2],clean_images_1.shape[3])
    else:
        raise NotImplementedError
    return intp_x
    

def get_interpolated_images_linear(noise_scheduler,clean_images_1,clean_images_2,t,n_steps):
    """
     Send tensors of same shape to this function. If length of shapes are 3 returns a tensor of size n_steps,3,H,W else if both tensors have a batch size of B
    returns a tensor of size B*n_steps,3,H,W. Interpolation is spherical
    
    """

    noisy_images_1 = forward_process(noise_scheduler,clean_images_1,t)
    noisy_images_2 = forward_process(noise_scheduler,clean_images_2,t)
    interpolated_images = []
    for i in range(n_steps):
        interpolated_images.append(noisy_images_1*(1-i/n_steps) + noisy_images_2*(i/n_steps))
    interpolated_images = torch.cat(interpolated_images,dim=0)
    d1 = len(clean_images_1.shape)
    d2 = len(clean_images_2.shape)

    if d1 == 3 and d2 == 3:
        interpolated_images = interpolated_images.view(-1,3,clean_images_1.shape[1],clean_images_1.shape[2])
    elif d1 == 4 and d2 == 4:
        interpolated_images = interpolated_images.view(-1,3,clean_images_1.shape[2],clean_images_1.shape[3])
    else:
        raise NotImplementedError
    return interpolated_images



def get_random_images_timestep_residual(n_samples,im_dim,noise_scheduler,model,t):

    """

    Get samples from a residual based noise scheduler denoised to a given time t

    Provide number of samples, image dimension, noise scheduler, model and time t to get samples from a residual based noise scheduler denoised to a given time t
    
    
    """

    rand_seed = torch.randn((n_samples,3,im_dim,im_dim)).to(model.device)

    for time in range(noise_scheduler.num_training_steps-1,t,-1):
        with torch.no_grad():
            residual = model(rand_seed, time).sample

        rand_seed = noise_scheduler.step(residual, t, rand_seed).prev_sample

    return rand_seed








def extract_feature_noised_input(model,input,timestep):
  """
  Custom function to get bottleneck activations from Hugging Face UNet 2D Bottleneck. Timestep value is
  a long tensor of size (BS,) with a single value for each image in the batch. Input is an image that is 
  noised at timestep t.

  """
  activation = {}
  name = 'bottleneck'
  def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
  h = model.mid_block.register_forward_hook(getActivation(name))
  _ = model(input,timestep)[0]
  h.remove()
  return activation['bottleneck']


    




def calculate_gram_matrix(features):
    features = features.view(features.size(0),-1)
    G = torch.ones((features.shape[0],features.shape[0]), device=features.device)
    for i in range(features.shape[0]):
        for j in range(features.shape[0]):
            G[i,j] = torch.dot(features[i],features[j])/(torch.norm(features[i])*torch.norm(features[j]))
    return G

def calculate_effective_rank(G):
    U, S, Vh = torch.linalg.svd(G)
    S = S/torch.sum(S)
    sum = 0.0
    for i in range(len(S)):
        sum += S[i]*torch.log(S[i])
    return -1.0*(sum.item())

def calculate_effective_rank_from_features(features):
    G = calculate_gram_matrix(features)
    return calculate_effective_rank(G)
