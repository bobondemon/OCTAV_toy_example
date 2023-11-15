import numpy as np
import matplotlib.pyplot as plt


def plot_weight_hist(weight17, weight45):
    w17min, w17max = np.min(weight17), np.max(weight17)
    w45min, w45max = np.min(weight45), np.max(weight45)
    # plot the weights histogram
    fontsize = 12
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(weight17, bins=100, linewidth=0.5, edgecolor="white", density=True)
    axs[0].set_title("Layer #17;\n(wmin, wmax) = ({:.2}, {:.2})".format(w17min, w17max))
    axs[0].set_xlabel("weight value", fontsize=fontsize)
    axs[0].set_ylabel("probability mass (%)", fontsize=fontsize)
    axs[0].grid(True)
    axs[1].hist(weight45, bins=100, linewidth=0.5, edgecolor="white", density=True)
    axs[1].set_title("Layer #45;\n(wmin, wmax) = ({:.2}, {:.2})".format(w45min, w45max))
    axs[1].set_xlabel("weight value", fontsize=fontsize)
    axs[1].set_ylabel("probability mass (%)", fontsize=fontsize)
    axs[1].grid(True)
    fig.suptitle("Weights histogram in probability mass")
    fig.tight_layout()
    plt.show()


def plot_qerror_hist_layer_17_45(qstepsize_17, qerrors17, qstepsize_45, qerrors45):
    print(f"qstepsize_17={qstepsize_17}; qstepsize_45={qstepsize_45}")
    q17min, q17max = np.min(qerrors17), np.max(qerrors17)
    q45min, q45max = np.min(qerrors45), np.max(qerrors45)
    print(f"(q17min, q17max)=({q17min}, {q17max}); (q45min, q45max)=({q45min}, {q45max})")
    assert (q17min > -qstepsize_17 / 2) and (q17max < qstepsize_17 / 2)
    assert (q45min >= -qstepsize_45 / 2) and (q45max <= qstepsize_45 / 2)
    fontsize = 12
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(qerrors17, bins=300, linewidth=0.5, edgecolor="white", density=True)
    axs[0].set_title("Layer #17;\n(qerror min, qerror max) = ({:.2}, {:.2})".format(q17min, q17max))
    axs[0].set_xlabel("qerror", fontsize=fontsize)
    axs[0].set_ylabel("probability mass (%)", fontsize=fontsize)
    axs[0].set_xlim([-qstepsize_17 / 2, qstepsize_17 / 2])
    axs[0].set_ylim([0, 100])
    axs[0].plot([-qstepsize_17 / 2, qstepsize_17 / 2], [1.0 / qstepsize_17] * 2, "r--", linewidth=4)
    axs[0].grid(True)
    axs[1].hist(qerrors45, bins=300, linewidth=0.5, edgecolor="white", density=True)
    axs[1].set_title("Layer #45;\n(qerror min, qerror max) = ({:.2}, {:.2})".format(q45min, q45max))
    axs[1].set_xlabel("qerror", fontsize=fontsize)
    axs[1].set_ylabel("probability mass (%)", fontsize=fontsize)
    axs[1].set_xlim([-qstepsize_45 / 2, qstepsize_45 / 2])
    axs[1].set_ylim([0, 100])
    axs[1].plot([-qstepsize_45 / 2, qstepsize_45 / 2], [1.0 / qstepsize_45] * 2, "r--", linewidth=4)
    axs[1].grid(True)
    fig.suptitle("`qerror` seems to be uniformly distributed in the range: " + r"$[-\Delta v/2, \Delta v/2]$")
    fig.tight_layout()
    plt.show()


def plot_qerror_layer_17_45(
    qerrors17_em,
    clipping_scalars17_em,
    qerrors45_em,
    clipping_scalars45_em,
    qerrors17_th,
    clipping_scalars17_th,
    qerrors45_th,
    clipping_scalars45_th,
    opt_Newton17,
    opt_Newton45,
):
    fontsize = 12
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(clipping_scalars17_em, qerrors17_em, "-b", clipping_scalars17_th, qerrors17_th, "--r")
    axs[0].plot(
        opt_Newton17[0], opt_Newton17[1], marker="o", markersize=7, markeredgecolor="red", markerfacecolor="green"
    )
    axs[0].text(opt_Newton17[0], opt_Newton17[1] * 2, "{:.4}".format(opt_Newton17[0]))
    axs[0].set_xlabel("clipping scalar", fontsize=fontsize)
    axs[0].set_ylabel("mse qerror", fontsize=fontsize)
    axs[0].set_title("layer #17", fontsize=fontsize)
    axs[0].legend(["empirical", "theoretical", "minimum by Newton"])
    axs[0].grid(True)

    axs[1].plot(clipping_scalars45_em, qerrors45_em, "-b", clipping_scalars45_th, qerrors45_th, "--r")
    axs[1].plot(
        opt_Newton45[0], opt_Newton45[1], marker="o", markersize=7, markeredgecolor="red", markerfacecolor="green"
    )
    axs[1].text(opt_Newton45[0], opt_Newton45[1] * 2, "{:.4}".format(opt_Newton45[0]))
    axs[1].set_xlabel("clipping scalar", fontsize=fontsize)
    axs[1].set_ylabel("mse qerror", fontsize=fontsize)
    axs[1].set_title("layer #45", fontsize=fontsize)
    axs[1].legend(["empirical", "theoretical", "minimum by Newton"])
    axs[1].grid(True)

    fig.suptitle("MSE Quantization Error")
    fig.tight_layout()
    plt.show()
