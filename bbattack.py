import numpy as np
import random
import torch
from PIL import Image
import math
from scipy.fftpack import dct, idct
def get_preds_(model, inputs, mean, std, correct_class=None, batch_size=50, return_cpu=True):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    all_preds, all_probs = None, None
    with torch.no_grad():
        for i in range(num_batches):
            upper = min((i + 1) * batch_size, inputs.size(0))
            input = apply_normalization_(inputs[(i * batch_size):upper], mean, std)
            input_var = input
            output = model(input_var)#.argmax(1)
            if correct_class is None:
                # this is top1 logit and pred
                prob, pred = output.max(1) # we only need hard labels
            else:
                prob, pred = output[:, correct_class], torch.autograd.Variable(torch.ones(output.size()) * correct_class)
            if return_cpu:
                prob = prob.data.cpu()
                pred = pred.data.cpu()
            else:
                prob = None
                pred = pred.data
            if i == 0:
                # if passing single image
                all_probs = prob
                all_preds = pred
            else:
                # if passing batch
                all_probs = torch.cat((all_probs, prob), 0)
                all_preds = torch.cat((all_preds, pred), 0)
    return all_preds, all_probs
# applies the normalization transformations
def apply_normalization_(imgs, mean, std):
    imgs_tensor = imgs.clone()
    if imgs.dim() == 2:
        imgs_tensor = (imgs_tensor - mean[0]) / std[0]
    elif imgs.dim() == 3:
        for i in range(imgs_tensor.size(0)):
            imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
    else:
        for i in range(imgs_tensor.size(1)):
            imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    return imgs_tensor
def lowbaAXAI(images,DATASET_MEAN, DATASET_STD, model,perturb_mode = 'dct',dct_ratio = 0.8,num_steps = 1234,spherical_step = 0.03,source_step = 0.01,repeat_images = 1,halve_every=250,blended_noise = True,batch_size=1):
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)

    trans = lambda x: x
    labels, _ = get_preds_(model, trans(images), mean=DATASET_MEAN, std=DATASET_STD, batch_size=batch_size)
    # print('labels:',labels)
    # sdfsdf
    if torch.cuda.is_available():
        labels = labels.cuda()
    # attack and save results
    images_batch = images
    labels_batch = labels
    all_perturbed = None
    all_mse_stats = None
    all_distance_stats = None
    all_spherical_step_stats = None
    all_source_step_stats = None

    perturbed, mse_stats, distance_stats, spherical_step_stats, source_step_stats = boundary_attack(
        model, images_batch, labels_batch, max_iters=num_steps, spherical_step=spherical_step,
        source_step=source_step, blended_noise=blended_noise, transformation=None, mean=DATASET_MEAN, std=DATASET_STD, dct_mode=(perturb_mode == 'dct'),
        dct_ratio=dct_ratio, repeat_images=repeat_images, halve_every=halve_every, verbose=False)
    if all_perturbed is None:
        all_perturbed = perturbed
        all_mse_stats = mse_stats
        all_distance_stats = distance_stats
        all_spherical_step_stats = spherical_step_stats
        all_source_step_stats = source_step_stats
    else:
        all_perturbed = torch.cat([all_perturbed, perturbed], dim=0)
        all_mse_stats = torch.cat([all_mse_stats, mse_stats], dim=0)
        all_distance_stats = torch.cat([all_distance_stats, distance_stats], dim=0)
        all_spherical_step_stats = torch.cat([all_spherical_step_stats, spherical_step_stats], dim=0)
        all_source_step_stats = torch.cat([all_source_step_stats, source_step_stats], dim=0)
    return all_perturbed

def sample_gaussian_torch(image_size, dct_ratio=1.0):
    x = torch.zeros(image_size)
    fill_size = int(image_size[-1] * dct_ratio)
    x[:, :, :fill_size, :fill_size] = torch.randn(x.size(0), x.size(1), fill_size, fill_size)
    if dct_ratio < 1.0:
        x = torch.from_numpy(idct(idct(x.numpy(), axis=3, norm='ortho'), axis=2, norm='ortho'))
    return x

def boundary_attack(
    model, images, labels, targeted=False, init=None, max_iters=1000, spherical_step=0.01,
    source_step=0.01, step_adaptation=1.5, reset_step_every=50, transformation=None,
    mean=None, std=None, blended_noise=False, dct_mode='none', dct_ratio=1.0,
    repeat_images=1, halve_every=250, verbose=True):    
    if transformation is None:
        transformation = lambda x: x
    
    batch_size = images.size(0)
    base_preds, _ = get_preds_(model, transformation(images), mean, std, batch_size=batch_size, return_cpu=False)
    images = images.repeat(repeat_images, 1, 1, 1)
    labels = labels.repeat(repeat_images)
    if repeat_images > 1:
        multipliers = (torch.ones(repeat_images) * 2).pow(torch.arange(0, repeat_images).float())
        dct_ratio = torch.ones(batch_size) * dct_ratio
        dct_ratio = (dct_ratio.unsqueeze(0).repeat(repeat_images, 1) * multipliers.unsqueeze(1).repeat(1, batch_size)).view(-1, 1).squeeze()
    images_vec = images.view(batch_size, -1)
    spherical_step_stats = torch.zeros(batch_size, max_iters)
    source_step_stats = torch.zeros(batch_size, max_iters)
    mse_stats = torch.zeros(batch_size, max_iters)
    distance_stats = torch.zeros(batch_size, max_iters)
    
    # sample random noise as initialization
    init = torch.zeros(images.size())#.cuda()
    if torch.cuda.is_available():
        init = init.cuda()
    preds = labels.clone()
    while preds.eq(labels).sum() > 0:
        if verbose==True:
            print("trying again")
        # idx = torch.arange(0, batch_size).long().cuda()[preds.eq(labels)]
        idx = torch.arange(0, batch_size).long()[preds.eq(labels)]
        if torch.cuda.is_available():
            idx = idx.cuda()
        noise = torch.rand(images[idx].size())

        if torch.cuda.is_available():
            noise = noise.cuda()
        init[idx] = noise#.cuda()
        # if torch.cuda.is_available():
        #     init[idx] = init[idx].cuda()

        preds, _ = get_preds_(model, transformation(init), mean, std, batch_size=batch_size, return_cpu=False)
    
    if blended_noise:

        min_alpha = torch.zeros(batch_size)#.cuda()
        max_alpha = torch.ones(batch_size)#.cuda()
        if torch.cuda.is_available():
            min_alpha = min_alpha.cuda()
            max_alpha = max_alpha.cuda()
        # binary search up to precision 2^(-10)
        for _ in range(10):
            alpha = (min_alpha + max_alpha) / 2
            alpha_expanded = alpha.view(batch_size, 1, 1, 1).expand_as(init)
            interp = alpha_expanded * init + (1 - alpha_expanded) * images
            preds, _ = get_preds_(model, transformation(interp), mean, std, batch_size=batch_size, return_cpu=False)
            if targeted:
                min_alpha[preds.ne(labels)] = alpha[preds.ne(labels)]
                max_alpha[preds.eq(labels)] = alpha[preds.eq(labels)]
            else:
                min_alpha[preds.eq(labels)] = alpha[preds.eq(labels)]
                max_alpha[preds.ne(labels)] = alpha[preds.ne(labels)]
        alpha = max_alpha.view(batch_size, 1, 1, 1).expand_as(init)
        perturbed = alpha * init + (1 - alpha) * images
    else:
        perturbed = init
        
    # recording success rate of previous moves for adjusting step size
    spherical_succ = torch.zeros(batch_size, reset_step_every)
    source_succ = torch.zeros(batch_size, reset_step_every)
    spherical_steps = (torch.ones(batch_size) * spherical_step)
    source_steps = (torch.ones(batch_size) * source_step)
    if torch.cuda.is_available():
        spherical_succ=spherical_succ.cuda()
        source_succ=source_succ.cuda()
        spherical_steps=spherical_steps.cuda()
        source_steps=source_steps.cuda()
    
    for i in range(max_iters):
        candidates, spherical_candidates = generate_candidate(
            images, perturbed, spherical_steps, source_steps, dct_mode=dct_mode, dct_ratio=dct_ratio)
        # additional query on spherical candidate for RGB-BA
        if dct_mode:
            spherical_preds = labels + 1
        else:
            spherical_preds, _ = get_preds_(model, transformation(spherical_candidates), mean, std, batch_size=batch_size, return_cpu=False)
        source_preds, _ = get_preds_(model, transformation(candidates), mean, std, batch_size=batch_size, return_cpu=False)
        spherical_succ[:, i % reset_step_every][spherical_preds.ne(labels)] = 1
        source_succ[:, i % reset_step_every][source_preds.ne(labels)] = 1
        # reject moves if they result in correctly classified images
        if source_preds.eq(labels).sum() > 0:
            idx = torch.arange(0, batch_size).long()[source_preds.eq(labels)]
            if torch.cuda.is_available():
                idx = idx.cuda()
            candidates[idx] = perturbed[idx]

        # reject moves if MSE is already low enough
        if i > 0:
            candidates[mse_prev.lt(1e-6)] = perturbed[mse_prev.lt(1e-6)]
        # record some stats
        perturbed_vec = perturbed.view(batch_size, -1)
        candidates_vec = candidates.view(batch_size, -1)
        mse_prev = (images_vec - perturbed_vec).pow(2).mean(1)
        mse = (images_vec - candidates_vec).pow(2).mean(1)
        reduction = 100 * (mse_prev.mean() - mse.mean()) / mse_prev.mean()
        norms = (images_vec - candidates_vec).norm(2, 1)
        if verbose==True:
            print('Iteration %d:  MSE = %.6f (reduced by %.4f%%), L2 norm = %.4f' % (i + 1, mse.mean(), reduction, norms.mean()))
        
        if (i + 1) % reset_step_every == 0:
            # adjust step size
            spherical_steps, source_steps, p_spherical, p_source = adjust_step(spherical_succ, source_succ, spherical_steps, source_steps, step_adaptation, dct_mode=dct_mode)
            spherical_succ.fill_(0)
            source_succ.fill_(0)
            if verbose==True:
                print('Spherical success rate = %.4f, new spherical step = %.4f' % (p_spherical.mean(), spherical_steps.mean()))
                print('Source success rate = %.4f, new source step = %.4f' % (p_source.mean(), source_steps.mean()))
            
        mse_stats[:, i] = mse
        distance_stats[:, i] = norms
        spherical_step_stats[:, i] = spherical_steps
        source_step_stats[:, i] = source_steps
        perturbed = candidates
        
        if halve_every > 0 and perturbed.size(0) > batch_size and (i + 1) % halve_every == 0:
            sdfsdfsdfsdfsdfsdfs
            # apply Hyperband to cut unsuccessful branches
            num_repeats = int(batch_size / batch_size)
            perturbed_vec = perturbed.view(batch_size, -1)
            mse = (images_vec - perturbed_vec).pow(2).mean(1).view(num_repeats, batch_size)
            _, indices = mse.sort(0)
            indices = indices[:int(num_repeats / 2)].cpu()
            idx = torch.arange(0.0, float(batch_size)).unsqueeze(0).repeat(int(num_repeats / 2), 1).long()
            idx += indices * batch_size
            idx = idx.view(-1, 1).squeeze()
            batch_size = idx.size(0)
            if torch.cuda.is_available():
                idx = idx.cuda()
            images = images[idx]
            labels = labels[idx]
            images_vec = images_vec[idx]
            perturbed = perturbed[idx]
            spherical_step_stats = spherical_step_stats[idx]
            source_step_stats = source_step_stats[idx]
            mse_stats = mse_stats[idx]
            distance_stats = distance_stats[idx]
            dct_ratio = dct_ratio[idx]
            spherical_steps = spherical_steps[idx]
            source_steps = source_steps[idx]
            spherical_succ = spherical_succ[idx]
            source_succ = source_succ[idx]
            
    return perturbed, mse_stats, distance_stats, spherical_step_stats, source_step_stats
  
    
def generate_candidate(images, perturbed, spherical_steps, source_steps, dct_mode='none', dct_ratio=1.0):
    
    batch_size = images.size(0)
    unnormalized_source_direction = images - perturbed
    source_norm = unnormalized_source_direction.view(batch_size, -1).norm(2, 1)
    source_direction = unnormalized_source_direction.div(source_norm.view(batch_size, 1, 1, 1).expand_as(unnormalized_source_direction))
    
    perturbation = sample_gaussian_torch(images.size(), dct_ratio=dct_ratio)
    if torch.cuda.is_available():
        perturbation = perturbation.cuda()
    
    if not dct_mode:
        dot = (images * perturbation).view(batch_size, -1).sum(1)
        perturbation -= source_direction.mul(dot.view(batch_size, 1, 1, 1).expand_as(source_direction))
    alpha = spherical_steps * source_norm / perturbation.view(batch_size, -1).norm(2, 1)
    perturbation = perturbation.mul(alpha.view(batch_size, 1, 1, 1).expand_as(perturbation))
    if not dct_mode:
        D = spherical_steps.pow(2).add(1).pow(-0.5)
        direction = perturbation - unnormalized_source_direction
        spherical_candidates = (images + direction.mul(D.view(batch_size, 1, 1, 1).expand_as(direction)))
    else:
        spherical_candidates = perturbed + perturbation
    spherical_candidates = spherical_candidates.clamp(0, 1)
    
    new_source_direction = images - spherical_candidates
    new_source_direction_norm = new_source_direction.view(batch_size, -1).norm(2, 1)
    length = source_steps * source_norm
    deviation = new_source_direction_norm - source_norm
    length += deviation
    length[length.le(0)] = 0
    length = length / new_source_direction_norm
    candidates = (spherical_candidates + new_source_direction.mul(length.view(batch_size, 1, 1, 1).expand_as(new_source_direction)))
    candidates = candidates.clamp(0, 1)
    
    return (candidates, spherical_candidates)


def adjust_step(spherical_succ, source_succ, spherical_steps, source_steps, step_adaptation, dct_mode='none'):
    p_spherical = spherical_succ.mean(1)
    num_spherical = spherical_succ.sum(1)
    p_source = torch.zeros(source_succ.size(0))#.cuda()
    if torch.cuda.is_available():
        p_source=p_source.cuda()
    for i in range(source_succ.size(0)):
        if num_spherical[i] == 0:
            p_source[i] = 0
        else:
            p_source[i] = source_succ[i, :][spherical_succ[i].eq(1)].mean()
    if not dct_mode:
        # adjust spherical steps when using RGB-BA
        spherical_steps[p_spherical.lt(0.2)] = spherical_steps[p_spherical.lt(0.2)] / step_adaptation
        spherical_steps[p_spherical.gt(0.6)] = spherical_steps[p_spherical.gt(0.6)] * step_adaptation
    source_steps[num_spherical.ge(10) * p_source.lt(0.2)] = source_steps[num_spherical.ge(10) * p_source.lt(0.2)] / step_adaptation
    source_steps[num_spherical.ge(10) * p_source.gt(0.6)] = source_steps[num_spherical.ge(10) * p_source.gt(0.6)] * step_adaptation
    return (spherical_steps, source_steps, p_spherical, p_source)


