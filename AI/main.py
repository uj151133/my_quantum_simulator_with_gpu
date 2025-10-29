import argparse
from libs.cfg import Config
from libs.train.supervised import run_supervised
from libs.train.ppo import run_ppo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sl", "ppo"], default="ppo",
                        help="sl: supervised imitation, ppo: reinforcement learning")
    parser.add_argument("--device", default=None, help="override device (cpu/cuda)")
    args = parser.parse_args()

    cfg = Config()
    if args.device:
        cfg.device = args.device

    if args.mode == "sl":
        run_supervised(cfg)
    else:
        run_ppo(cfg)

if __name__ == "__main__":
    main()