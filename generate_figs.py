import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from ntp.common import detach
from ntp.implicit_burgers_sol import compute_exact_implicit
from ntp.mpl_rc import update_rcParams

def plot_forward_backward(data: dict, key: str, save_path: str):
    x_data = list(data.keys())
    x_data.remove('info')
    x_data.sort()

    ntp_times = [sum(data[i][key]['ntp']) / len(data[i][key]['ntp']) for i in x_data]
    ad_times = [sum(data[i][key]['ad']) / len(data[i][key]['ad']) for i in x_data]

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    for ax in axs:
        ax.plot(x_data, ntp_times, label='$n$-TangentProp', marker='*', markeredgecolor='black', markersize=15, linestyle='-', c='b')
        ax.plot(x_data, ad_times, label='Autodifferentiation', marker='*', markeredgecolor='black', markersize=15, linestyle='--', c='r')

    axs[1].set_yscale('log')
    axs[0].set_ylabel("Average Runtime")
    axs[0].set_ylabel("Average Runtime (log Scale)")
    axs[1].set_xlabel("Number of Derivatives")
    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{key}_derivs.png'), bbox_inches='tight')

def plot_full_explore(data: dict, key: str, save_path: str):
    depths = set([k[0] for k in data.keys()])
    widths = set([k[1] for k in data.keys()])
    batch_sizes = set([k[2] for k in data.keys()])
    derivs = set([k[3] for k in data.keys()])
    modes = set([k[4] for k in data.keys()])

    depths_map = {d: i for i, d in enumerate(sorted(depths))}
    derivs_map = {d: i for i, d in enumerate(sorted(derivs))}

    grid = [[{b: {w: {m: {} for m in modes} for w in widths} for b in batch_sizes} for _ in range(len(derivs))] for _ in range(len(depths))]

    for k, v in data.items():
        d, w, b, der, m = k
        grid[depths_map[d]][derivs_map[der]][b][w][m]['ttime'] = sum(v['ttime']) / len(v['ttime'])
        grid[depths_map[d]][derivs_map[der]][b][w][m]['ftime'] = sum(v['ftime']) / len(v['ftime'])
        grid[depths_map[d]][derivs_map[der]][b][w][m]['btime'] = sum(v['btime']) / len(v['btime'])

        grid[depths_map[d]][derivs_map[der]][b][w][m]['ttime_std'] = torch.std(torch.tensor(v['ttime']))
        grid[depths_map[d]][derivs_map[der]][b][w][m]['ftime_std'] = torch.std(torch.tensor(v['ftime']))
        grid[depths_map[d]][derivs_map[der]][b][w][m]['btime_std'] = torch.std(torch.tensor(v['btime']))

    sorted_depths = sorted(list(depths))
    sorted_derivs = sorted(list(derivs))

    fig, axs = plt.subplots(4, 3, figsize=(8, 12), dpi=300)

    for j, g in enumerate(grid):
        for i, gg in enumerate(g):
            gg = {k: v for k, v in sorted(gg.items(), key=lambda x: x[0])}
            for bs, plot_data in gg.items():
                if bs != 64 and bs != 256:
                    x = list(plot_data.keys())
                    y_ad = [plot_data[k]['ad'][key] for k in x]
                    y_ntp = [plot_data[k]['ntp'][key] for k in x]

                    y = [ad / ntp for ad, ntp in zip(y_ad, y_ntp)]

                    # sort x and y according to x
                    x, y = zip(*sorted(zip(x, y)))

                    axs[i][j].plot(x, y, label=f'bs={bs}', marker='*',  markeredgecolor='black', markersize=8)
                    if i == 0:
                        axs[i][j].set_title(f'Depth {sorted_depths[j]}')
                    if i == 3:
                        axs[i][j].set_xlabel('Width')
                    if j == 0:
                        axs[i][j].set_ylabel('ad / ntp Ratio')
                        axs[i][j].text(-0.5, 0.5, f'{sorted_derivs[i]} Derivatives', fontsize=15, ha='right', va='center', transform=axs[i][j].transAxes, rotation=65)

                    if i == 0 or i == 1:
                        axs[i][j].set_ylim([0., 2.5])
                    elif i == 2:
                        axs[i][j].set_ylim([0., 5.])
                    else:
                        axs[i][j].set_ylim([0., 20.])

                    axs[i][j].axhline(1, color='k', linestyle='--', label='ad / ntp ratio = 1')
                    # axs[i][j].legend()

    handles, labels = axs[0][0].get_legend_handles_labels()
    handles = handles[:-1:2] + [handles[-1]]
    labels = labels[:-1:2] + [labels[-1]]
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{key}_full_grid.png'), bbox_inches='tight')

def plot_profile(
    sol: torch.Tensor, 
    losses:dict[str, list[int]], 
    true_lambda: float,
    profile_name: str,
):
    n_derivs = len(sol)
    n_rows = n_derivs // 2 + 2

    Uderivs = compute_exact_implicit(
        torch.linspace(-1, 1, 100).reshape(-1, 1),
        m=true_lambda,
        a=-2.,
        A=1.,
        n_derivs=n_derivs
    )

    X = Uderivs[0]
    Uderivs = Uderivs[1:]
    x = torch.linspace(-2., 2., sol[0].shape[0])

    fig = plt.figure(figsize=(10, 3 * n_rows))
    gs = gridspec.GridSpec(n_rows, 2, height_ratios=([2/3] * (n_rows - 2)) + [1, 1])

    axs = []
    row = 0
    for _ in range(n_rows - 2):
        axs.append(fig.add_subplot(gs[row, 0]))
        axs.append(fig.add_subplot(gs[row, 1]))
        row += 1

    axs.append(fig.add_subplot(gs[-2, :]))
    axs.append(fig.add_subplot(gs[-1, :]))

    deriv_names = [
        'Profile', 'First Derivative', 'Second Derivative', 'Third Derivative',
        'Fourth Derivative', 'Fifth Derivative', 'Sixth Derivative',
        'Seventh Derivative', 'Eighth Derivative', 'Ninth Derivative'
    ]

    for i in range(n_derivs):
        axs[i].plot(detach(X), detach(Uderivs[i]), c='b', linestyle='-', linewidth=3)
        axs[i].plot(detach(x), detach(sol[i]), c='r', linestyle='--', linewidth=3)
        axs[i].set_title(deriv_names[i])

    axs[-2].plot(losses['loss'][10:], c='k')
    axs[-2].set_ylabel('Loss')
    axs[-2].set_yscale('log')

    axs[-1].plot([abs(r - true_lambda) for r in losses['r'][10:]], c='k')
    axs[-1].set_yscale('log')
    axs[-1].set_xlabel('Epochs')
    axs[-1].set_ylabel(f"$|\lambda - {true_lambda}|$")

    plt.tight_layout()
    plt.savefig(f"{profile_name}.png")

def profile_ntp_vs_ad(rate1: list[float], rate2: list[float], save_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(rate1, color='blue', label="First Profile")
    plt.plot(rate2, color='red', label="Second Profile")
    plt.axhline(1., color='k', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Ratio of Autodiff to $n$-TangentProp')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'rate_comparison.png'))

if __name__ == "__main__":
    update_rcParams()

    # Generate the figures from the forward-backwards passes
    forward_backward_data = torch.load('results/scale_with_derivs/scale_with_derivs.pt')

    for key in ['total_time', 'forward_time', 'backward_time']:
        plot_forward_backward(forward_backward_data, key, 'fig')
        
    # Generate the figures summarizing the full exploration
    full_explore_data = torch.load('results/full_explore/full_explore.pt')

    plot_full_explore(full_explore_data, 'ttime', 'fig')
    plot_full_explore(full_explore_data, 'ftime', 'fig')

    # Generate the figures summarizing the self-similar Burgers runs
    profile_bases = [
        'results/profiles/profile_0',
        'results/profiles/profile_1',
        'results/profiles/profile_2',
        'results/profiles/profile_3',
    ]

    sols = [
        torch.load(base + '_ntp_out.pt')[0] for base in profile_bases
    ]

    losses = [
        torch.load(base + '_ntp_losses.pt') for base in profile_bases
    ]

    true_lambdas = [1 / (2 * k + 2) for k in range(len(sols))]

    for i, (sol, loss, true_lambda) in enumerate(zip(sols, losses, true_lambdas)):
        plot_profile(sol, loss, true_lambda, f"fig/profile_{i + 1}")

    # Generate rate comparasion figure
    ntp_times_1 = torch.load('results/profiles/profile_0_ntp_times.pt')
    ad_times_1 = torch.load('results/profiles/profile_0_ad_times.pt')

    ntp_times_2 = torch.load('results/profiles/profile_1_ntp_times.pt')
    ad_times_2 = torch.load('results/profiles/profile_1_ad_times.pt')

    rate_1 = [ad / ntp for ad, ntp in zip(ad_times_1, ntp_times_1)]
    rate_2 = [ad / ntp for ad, ntp in zip(ad_times_2, ntp_times_2)]

    profile_ntp_vs_ad(rate_1, rate_2, 'fig')