import torch as th
import sys
import numpy as np

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)


def p_epsilon_var(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)
        min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {
            "eps": model_output,
            "mean": model_mean,
            "log_variance": model_log_variance,
        }

def p_epsilon(self, model, x, t, model_kwargs=None):
    """
    Apply the model to get p(x_{t-1} | x_t)

    :param model: the model, which takes a signal and a batch of timesteps
                    as input.
    :param x: the [N x C x ...] tensor at time t.
    :param t: a 1-D Tensor of timesteps.
    :param clip_denoised: if True, clip the denoised signal into [-1, 1].
    :param denoised_fn: if not None, a function which applies to the
        x_start prediction before it is used to sample. Applies before
        clip_denoised.
    :param model_kwargs: if not None, a dict of extra keyword arguments to
        pass to the model. This can be used for conditioning.
    :return: a dict with the following keys:
                - 'eps': the model epsilon output.
                - 'var': the model variance output.
    """

    if model_kwargs is None:
        model_kwargs = {}

    B, C = x.shape[:2]
    assert t.shape == (B,)
    model_output = model(x, t, **model_kwargs)

    assert model_output.shape == (B, C * 2, *x.shape[2:])
    model_output, model_var_values = th.split(model_output, C, dim=1)
    
    return {'eps': model_output, 'var': model_var_values}

def returnSample_CF(model, model_fn, model_kwargs, device, batch_size, options, diffusion, clip_denoised=True, noise=None, progress=True, cond_fn=None):

    full_batch_size = batch_size * 2

    model.del_cache()
    samples_CF = diffusion.p_sample_loop(
        model_fn,
        (full_batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=clip_denoised,
        noise=noise,
        progress=progress,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
    )
    model.del_cache()
    return samples_CF

def returnSample_CLIP(model, diffusion, batch_size, device, model_kwargs, options, cond_fn=None):
    model.del_cache()
    samples_CLIP = diffusion.p_sample_loop(
        model,
        (batch_size, 3, options["image_size"], options["image_size"]),
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
    )
    model.del_cache()

    return samples_CLIP

def returnUpSample(model_up, diffusion_up, batch_size, device, model_kwargs, options_up, upsample_temp=1.0, cond_fn=None, progress=True):
    model_up.del_cache()
    up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
    up_samples = diffusion_up.ddim_sample_loop(
        model_up,
        up_shape,
        noise=th.randn(up_shape, device=device) * upsample_temp,
        device=device,
        clip_denoised=True,
        progress=progress,
        model_kwargs=model_kwargs,
        cond_fn=cond_fn,
    )
    model_up.del_cache()

    return up_samples