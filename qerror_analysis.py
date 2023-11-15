import numpy as np
import torch
import matplotlib.pyplot as plt

from load_model_helper import load_pretrained_model_and_save, load_params_and_get_weights_17_45
from plot_helper import plot_weight_hist, plot_qerror_hist_layer_17_45, plot_qerror_layer_17_45


def cal_max_quant_stepsize(w, bit_num):
    # note that we assume zero_point is always 0 (symmetric quant.)
    w_max = np.max(np.abs(w))
    return w_max / 2 ** (bit_num - 1)


def theoretical_mse_qerror(w, clipping_scalar, bit_num, bins=500):
    hist, bin_edges = np.histogram(np.abs(w), bins=bins, density=False)
    hist = hist / np.sum(hist)  # turn into probability mass (note that it is different with density)

    clip_start_idx = np.where(np.diff(bin_edges > clipping_scalar))[0]
    clip_start_idx = 0 if len(clip_start_idx) == 0 else clip_start_idx[0]

    J1 = np.sum(hist[:clip_start_idx]) * (clipping_scalar**2 / (3 * 4**bit_num))
    J2 = 0.0
    for idx in range(clip_start_idx, len(hist)):
        prob_x_mass = hist[idx]
        x = (bin_edges[idx + 1] + bin_edges[idx]) / 2
        J2 += (clipping_scalar - x) ** 2 * prob_x_mass

    return J1 + J2


def cal_qerror(w, qstepsize, zero, bit_num):
    quant_min, quant_max = -(2 ** (bit_num - 1)), 2 ** (bit_num - 1) - 1
    w_q = torch.fake_quantize_per_tensor_affine(torch.as_tensor(w), qstepsize, zero, quant_min, quant_max).numpy()
    return w - w_q


def do_empirical_qerror_scanning(w, qstepsize, zero, bit_num=4, scalar_num=200, plot_ratio=7.0):
    # `qstepsize` stands for quantization step size
    qerrors = []
    clipping_scalars = np.linspace(1e-2, qstepsize * plot_ratio, scalar_num)
    # for loop for each clipping scalar
    for cs in clipping_scalars:
        qerror = cal_qerror(w, 2 * cs / 2**bit_num, zero, bit_num)
        qerrors.append(np.mean(qerror**2))

    return qerrors, clipping_scalars


def do_theoretical_qerror_calculation(w, qstepsize, bit_num, scalar_num=200, plot_ratio=7.0):
    qerrors = []
    clipping_scalars = np.linspace(1e-2, qstepsize * plot_ratio, scalar_num)
    # for loop for each clipping scalar
    for cs in clipping_scalars:
        qerrors.append(theoretical_mse_qerror(w, cs, bit_num))
    return qerrors, clipping_scalars


def find_opt_by_Newton_method(weights, bit_num, cs_init=0.0, iter_num=10):
    # `cs` stands for `clipping scalar`
    weights_abs = np.abs(weights)
    cs_cur = cs_init
    for itr in range(iter_num):
        indicator_larger = weights_abs > cs_cur
        indicator_smaller = weights_abs <= cs_cur  # should we ignore case with `==0`?
        numerator = np.sum(weights_abs[indicator_larger])
        denominator = np.sum(indicator_smaller) / (3 * 4**bit_num) + np.sum(indicator_larger)
        cs_cur = numerator / denominator
    return cs_cur


def run():
    mdl_path = "./model_params/resnet50_model.pkl"
    load_pretrained_model_and_save(mdl_path, resnet_version=50)
    weight17, weight45 = load_params_and_get_weights_17_45(mdl_path)

    plot_weight_hist(weight17, weight45)

    # quantization config
    bit_num = 4
    zero = 0
    plot_ratio = 7.0
    # `qstepsize` stands for quantization step size
    qstepsize_17, qstepsize_45 = cal_max_quant_stepsize(weight17, bit_num), cal_max_quant_stepsize(weight45, bit_num)

    # plot qerror histogram
    plot_qerror_hist_layer_17_45(
        qstepsize_17,
        cal_qerror(weight17, qstepsize_17, zero, bit_num),
        qstepsize_45,
        cal_qerror(weight45, qstepsize_45, zero, bit_num),
    )

    # do empirical qerror scanning
    qerrors17, clipping_scalars = do_empirical_qerror_scanning(
        weight17, qstepsize_17, zero, bit_num, plot_ratio=plot_ratio
    )
    qerrors45, clipping_scalars = do_empirical_qerror_scanning(
        weight45, qstepsize_45, zero, bit_num, plot_ratio=plot_ratio
    )

    # do theoretical qerror calculation
    theoretical_qerrors17, theoretical_clipping_scalars = do_theoretical_qerror_calculation(
        weight17, qstepsize_17, bit_num, plot_ratio=plot_ratio
    )
    theoretical_qerrors45, theoretical_clipping_scalars = do_theoretical_qerror_calculation(
        weight45, qstepsize_45, bit_num, plot_ratio=plot_ratio
    )

    # find minimum with Newton's method
    opt_Newton_cs17 = find_opt_by_Newton_method(weight17, bit_num, cs_init=0.0)
    opt_Newton_mse17 = theoretical_mse_qerror(weight17, opt_Newton_cs17, bit_num)
    opt_Newton_cs45 = find_opt_by_Newton_method(weight45, bit_num, cs_init=0.0)
    opt_Newton_mse45 = theoretical_mse_qerror(weight45, opt_Newton_cs45, bit_num)

    # plot
    plot_qerror_layer_17_45(
        qerrors17,
        clipping_scalars,
        qerrors45,
        clipping_scalars,
        theoretical_qerrors17,
        theoretical_clipping_scalars,
        theoretical_qerrors45,
        theoretical_clipping_scalars,
        [opt_Newton_cs17, opt_Newton_mse17],
        [opt_Newton_cs45, opt_Newton_mse45],
    )


if __name__ == "__main__":
    run()
