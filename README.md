# ctc-contest

Each experiment has its own directory with `config.txt`, `metrics.csv`, checkpoints and logs.

```powershell
python main.py --mode train --batch-size 32 --epochs 20 --lr 3e-4 --checkpoint-dir runs/exp001
python main.py --mode predict --checkpoint-dir runs/exp001 --submission runs/exp001/submission.csv
```

Track `config.txt` and `metrics.csv` in Git. Checkpoints, logs and submissions are ignored.
