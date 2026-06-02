import argparse
import csv
import random
import wave
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset


CHARS = "-0123456789"
BLANK = 0
char_to_id = {char: index + 1 for index, char in enumerate(CHARS)}
id_to_char = {index + 1: char for index, char in enumerate(CHARS)}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_wav(path):
    with wave.open(str(path), "rb") as file:
        if file.getframerate() != 8000 or file.getnchannels() != 1 or file.getsampwidth() != 2:
            raise ValueError()

        raw = bytearray(file.readframes(file.getnframes()))

    return torch.frombuffer(raw, dtype=torch.int16).float() / 32768.0


class MorseDataset(Dataset):
    def __init__(self, rows, audio_dir, augment=False):
        self.rows = rows
        self.audio_dir = Path(audio_dir)
        self.augment = augment
        self.window = torch.hann_window(256)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        wav = read_wav(self.audio_dir / row["filename"])
        if self.augment:
            wav = wav * random.uniform(0.85, 1.15)
            if random.random() < 0.5:
                wav = wav + torch.randn_like(wav) * random.uniform(0.0, 0.012)

        spec = torch.stft(
            wav,
            n_fft=256,
            hop_length=80,
            win_length=256,
            window=self.window,
            center=False,
            return_complex=True,
        ).abs()

        spec = torch.log1p(spec).transpose(0, 1)
        spec = (spec - spec.mean()) / (spec.std() + 1e-5)
        text = row.get("text")

        if text:
            target = torch.tensor([char_to_id[char] for char in text], dtype=torch.long)
        else:
            target = None

        return spec, target, row["filename"]


def collate(batch):
    specs, targets, filenames = zip(*batch)
    input_lengths = torch.tensor([len(spec) for spec in specs], dtype=torch.long)
    specs = pad_sequence(specs, batch_first=True)

    if targets[0] is None:
        return specs, input_lengths, None, None, filenames

    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    targets = torch.cat(targets)

    return specs, input_lengths, targets, target_lengths, filenames


class MorseCTC(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(129, 192, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Conv1d(192, 192, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(
            input_size=192,
            hidden_size=192,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(384, len(CHARS) + 1)

    def forward(self, specs, lengths):
        x = self.conv(specs.transpose(1, 2)).transpose(1, 2)
        lengths = (lengths + 1) // 2
        lengths = (lengths + 1) // 2
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.rnn(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True)
        return self.classifier(x).log_softmax(dim=-1).transpose(0, 1), lengths


def decode(log_probs, lengths):
    best = log_probs.argmax(dim=-1).transpose(0, 1).cpu()
    answers = []
    for ids, length in zip(best, lengths):
        previous = BLANK
        text = []
        for token in ids[: int(length)]:
            token = int(token)
            if token != BLANK and token != previous:
                text.append(id_to_char[token])
            previous = token
        answers.append("".join(text))
    return answers


def levenshtein(left, right):
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def load_state(path, model, optimizer=None, device="gpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def save_state(path, model, optimizer, epoch, score):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "score": score,
            "chars": CHARS,
        },
        path,
    )


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    distances = []
    examples = []
    for specs, input_lengths, targets, target_lengths, _ in loader:
        specs = specs.to(device)
        log_probs, output_lengths = model(specs, input_lengths)
        predictions = decode(log_probs, output_lengths)
        offset = 0
        for prediction, length in zip(predictions, target_lengths):
            length = int(length)
            truth = "".join(id_to_char[int(token)] for token in targets[offset: offset + length])
            offset += length
            distances.append(levenshtein(prediction, truth))
            if len(examples) < 3:
                examples.append((truth, prediction))
    return sum(distances) / len(distances), examples


def make_loaders(args):
    rows = read_csv(args.data_dir / "train" / "labels.csv")
    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    val_size = max(1, round(len(indices) * args.val_fraction))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_data = Subset(MorseDataset(rows, args.data_dir / "train", augment=True), train_indices)
    val_data = Subset(MorseDataset(rows, args.data_dir / "train"), val_indices)
    common = {"num_workers": args.workers, "collate_fn": collate, "pin_memory": torch.cuda.is_available()}
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, **common)
    return train_loader, val_loader


def train(args, device):
    train_loader, val_loader = make_loaders(args)
    model = MorseCTC().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    last_path = args.checkpoint_dir / "last.pt"
    best_path = args.checkpoint_dir / "best.pt"
    start_epoch = 0
    best_score = float("inf")
    if last_path.exists() and not args.fresh:
        checkpoint = load_state(last_path, model, optimizer, device)
        start_epoch = checkpoint["epoch"]
        best_score = checkpoint["score"]
        print(f"resume={last_path} completed_epochs={start_epoch} best_val={best_score:.4f}")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, (specs, input_lengths, targets, target_lengths, _) in enumerate(train_loader, start=1):
            specs = specs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            log_probs, output_lengths = model(specs, input_lengths)
            loss = loss_fn(log_probs, targets, output_lengths, target_lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach())
            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={total_loss / step:.4f}")
        score, examples = validate(model, val_loader, device)
        print(f"epoch={epoch} train_loss={total_loss / len(train_loader):.4f} val_levenshtein={score:.4f}")
        print("examples:", examples)
        if score < best_score:
            best_score = score
            save_state(best_path, model, optimizer, epoch, best_score)
            print(f"saved={best_path}")
        save_state(last_path, model, optimizer, epoch, best_score)
    return best_path


@torch.inference_mode()
def predict(args, device):
    best_path = args.checkpoint_dir / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No checkpoint: {best_path}. Run --mode train first.")
    rows = read_csv(args.sample_submission)
    dataset = MorseDataset([{"filename": row["filename"]} for row in rows], args.data_dir / "test")
    loader = DataLoader(
        dataset,
        batch_size=args.predict_batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    model = MorseCTC().to(device)
    checkpoint = load_state(best_path, model, device=device)
    if checkpoint.get("chars") != CHARS:
        raise ValueError()
    model.eval()
    predictions = []
    for step, (specs, input_lengths, _, _, _) in enumerate(loader, start=1):
        log_probs, output_lengths = model(specs.to(device, non_blocking=True), input_lengths)
        predictions.extend(decode(log_probs, output_lengths))
        if step % args.log_every == 0:
            print(f"predict step={step}/{len(loader)}")
    with open(args.submission, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "text"])
        writer.writeheader()
        for row, text in zip(rows, predictions):
            writer.writerow({"filename": row["filename"], "text": text})
    print(f"saved={args.submission} rows={len(predictions)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "train", "predict"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("morse_dataset"))
    parser.add_argument("--sample-submission", type=Path, default=Path("sample_submission.csv"))
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--predict-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if args.mode == "train":
        train(args, device)
    elif args.mode == "predict":
        predict(args, device)
    elif (args.checkpoint_dir / "best.pt").exists() and not args.fresh:
        print("best checkpoint already exists, skip training")
        predict(args, device)
    else:
        train(args, device)
        predict(args, device)


if __name__ == "__main__":
    main()
