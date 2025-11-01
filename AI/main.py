import argparse
from libs.cfg import Config
from libs.train.supervised import run_supervised
from libs.train.ppo import run_ppo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sl", "ppo"], default="ppo")
    parser.add_argument("--init-ckpt", default=None, help="PPO warm start checkpoint (.pt)")
    parser.add_argument("--no-export", action="store_true", help="disable final TorchScript/ONNX export")
    args = parser.parse_args()

    cfg = Config()  # すべての設定は cfg.py にハードコード

    if args.mode == "sl":
        run_supervised(cfg, export_final=not args.no_export)
    else:
        run_ppo(cfg, init_ckpt=args.init_ckpt, export_final=not args.no_export)

if __name__ == "__main__":
    main()