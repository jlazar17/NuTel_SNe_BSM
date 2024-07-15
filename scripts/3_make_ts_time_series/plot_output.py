import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

def initialize_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument(
        "--infile",
        type=str,
        required=True
    )
    parser.add_argument(
        "--trials_file",
        type=str,
        default=""
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=-1
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=15
    )
    parser.add_argument(
        "--bsm_names",
        nargs="+"
    )
    parser.add_argument(
        "--outfile",
        required=True,
        type=str
    )
    args = parser.parse_args()
    return args

def main(args=None) -> None:
    if args is None:
        args = initialize_args()

    if args.bsm_names is None:
        bsm_names = []

    fig, ax = plt.subplots()
    with h5.File(args.infile) as h5f:
        for v in h5f.values():
            if bsm_names and v.attrs["bsm_name"] not in bsm_names:
                continue
            times = v["times"][:]
            mask = np.logical_and(args.tmin <= times, times <= args.tmax)
            times = times[mask]
            test_stats = np.median(np.cumsum(v["test_statistic_series"][mask, :], axis=0), axis=1)

            line = ax.plot(times, test_stats, label=v.attrs["bsm_name"], lw=2)
            if not args.trials_file:
                continue
            with h5.File(args.trials_file) as h5f2:
                for v2 in h5f2.values():
                    if v2.attrs["bsm_name"]==v.attrs["bsm_name"]:
                        break
                    
                if v2.attrs["bsm_name"]!=v.attrs["bsm_name"]:
                    continue
                ax.axhline(np.quantile(v2["background/ts"], 0.95), color=line[0].get_color(), lw=1, ls="--")


    plt.xlim(args.tmin, args.tmax)
    plt.xlabel(r"$t~\left[\mathrm{s}\right]$")
    plt.ylabel(r"TS")
    plt.savefig(args.outfile)
    plt.close()

if __name__=="__main__":
    main()
