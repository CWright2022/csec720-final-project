# CSEC720 final project
# Fall 2026
# Cayden Wright, Eric Antonecchia

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
NUM_CLASSES = 43
SEED = 720
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'gtsrb_cnn.pth'
ADVERSARIAL_CACHE_DIR = ROOT / 'adversarial_cache'
MEAN = (0.3403, 0.3121, 0.3214)
STD = (0.2724, 0.2608, 0.2669)
EPSILONS = (0.0, 0.01, 0.03, 0.06)
PREPROCESSING = {'dataset': 'GTSRB', 'resize': [32, 32],
                 'mean': MEAN, 'std': STD, 'attack_space': 'raw_pixels_0_1'}
# Keep images in pixel space for attacks and later image transformations.
preprocess = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])


class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class PixelSpaceModel(nn.Module):
    """Accept raw pixels while preserving the original CNN's normalization."""
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.register_buffer('mean', torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.classifier((x - self.mean) / self.std)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loaders():
    train = datasets.GTSRB(root=str(ROOT / 'data'), split='train', download=True, transform=preprocess)
    test = datasets.GTSRB(root=str(ROOT / 'data'), split='test', download=True, transform=preprocess)
    train_size = int(0.8 * len(train))
    train, val = random_split(train, [train_size, len(train) - train_size],
                              generator=torch.Generator().manual_seed(SEED))
    def loader(dataset, shuffle):
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2,
                          worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(SEED))
    return loader(train, True), loader(val, False), loader(test, False)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total = 0.0, 0
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logits = model(inputs)
            total_loss += criterion(logits, labels).item() * len(labels)
            predictions = logits.argmax(1)
            pairs = (labels * NUM_CLASSES + predictions).cpu()
            confusion += torch.bincount(pairs, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
            total += len(labels)
    if not total:
        raise ValueError('Cannot evaluate an empty dataset')
    tp = confusion.diag().float()
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = torch.where(denominator > 0, 2 * tp / denominator.clamp_min(1), 0.0)
    return total_loss / total, tp.sum().item() / total, f1.mean().item()


def validate_attack_inputs(inputs, epsilon):
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError('epsilon must be finite and nonnegative')
    if not torch.isfinite(inputs).all() or inputs.min() < 0 or inputs.max() > 1:
        raise ValueError('Attacks require finite raw pixels in [0, 1]')


def fgsm_attack(model, inputs, labels, criterion, epsilon=0.03):
    model.eval()
    inputs = inputs.detach().clone().to(DEVICE)
    validate_attack_inputs(inputs, epsilon)
    if epsilon == 0:
        return inputs
    inputs.requires_grad_(True)
    loss = criterion(model(inputs), labels.to(DEVICE))
    gradient, = torch.autograd.grad(loss, inputs)
    return (inputs + epsilon * gradient.sign()).clamp(0, 1).detach()


def pgd_attack(model, inputs, labels, criterion, epsilon=0.03, alpha=0.005,
               iters=20, restarts=2, seed=SEED):
    model.eval()
    inputs = inputs.detach().clone().to(DEVICE)
    labels = labels.to(DEVICE)
    validate_attack_inputs(inputs, epsilon)
    if iters < 1 or restarts < 1 or alpha <= 0 or not np.isfinite(alpha):
        raise ValueError('PGD needs positive steps, restarts and finite step size')
    if epsilon == 0:
        return inputs
    generator = torch.Generator(device=inputs.device).manual_seed(seed)
    with torch.no_grad():
        best = inputs.clone()
        best_loss = nn.functional.cross_entropy(model(best), labels, reduction='none')
        best_success = model(best).argmax(1).ne(labels)

    # Retain successful examples and the strongest loss per image across every iterate.
    for _ in range(restarts):
        noise = torch.empty_like(inputs).uniform_(-epsilon, epsilon, generator=generator)
        adv = (inputs + noise).clamp(0, 1)
        for step in range(iters + 1):
            adv = adv.detach().requires_grad_(True)
            logits = model(adv)
            losses = nn.functional.cross_entropy(logits, labels, reduction='none')
            with torch.no_grad():
                success = logits.argmax(1).ne(labels)
                improve = (success & ~best_success) | ((success == best_success) & (losses > best_loss))
                best[improve] = adv.detach()[improve]
                best_loss[improve] = losses.detach()[improve]
                best_success[improve] = success[improve]
            if step == iters:
                break
            gradient, = torch.autograd.grad(losses.mean(), adv)
            adv = adv.detach() + alpha * gradient.sign()
            adv = torch.maximum(torch.minimum(adv, inputs + epsilon), inputs - epsilon).clamp(0, 1)
    return best.detach()


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        value = value.detach().cpu().contiguous()
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_or_generate_adversarial_dataset(model, loader, criterion, attack_name, attack_fn, attack_kwargs):
    """Cache exact ordered inputs, model identity, settings and clean predictions."""
    model.eval()
    clean_images, labels = [], []
    for batch, target in loader:
        clean_images.append(batch.cpu())
        labels.append(target.cpu())
    clean_images, labels = torch.cat(clean_images), torch.cat(labels)
    digest = hashlib.sha256(clean_images.numpy().tobytes())
    digest.update(labels.numpy().tobytes())
    metadata = {'version': 2, 'model': model_fingerprint(model), 'inputs': digest.hexdigest(),
                'preprocessing': PREPROCESSING, 'attack': attack_name, 'parameters': attack_kwargs,
                'batch_size': BATCH_SIZE, 'seed': SEED, 'torch': str(torch.__version__),
                'device': str(DEVICE), 'loss': 'cross_entropy'}
    # JSON roundtrip gives metadata the same types on save and reload.
    metadata = json.loads(json.dumps(metadata, sort_keys=True))
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    ADVERSARIAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = ADVERSARIAL_CACHE_DIR / f'v2_{attack_name}_{key}.pt'
    if cache_path.exists():
        cached = torch.load(cache_path, map_location='cpu', weights_only=True)
        if cached['metadata'] != metadata:
            raise ValueError('Adversarial cache metadata mismatch')
        print(f'Loading cached {attack_name.upper()} images: {cache_path.name}')
    else:
        images, clean_predictions = [], []
        paired = DataLoader(TensorDataset(clean_images, labels), batch_size=BATCH_SIZE, shuffle=False)
        for batch_index, (inputs, targets) in enumerate(paired):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            with torch.no_grad():
                clean_predictions.append(model(inputs).argmax(1).cpu())
            kwargs = dict(attack_kwargs)
            if attack_fn is pgd_attack:
                kwargs['seed'] = SEED + batch_index
            with torch.enable_grad():
                adversarial = attack_fn(model, inputs, targets, criterion, **kwargs)
            if (adversarial - inputs).abs().max().item() > attack_kwargs['epsilon'] + 1e-6:
                raise ValueError('Attack exceeded its pixel-space perturbation budget')
            images.append(adversarial.cpu())
        cached = {'images': torch.cat(images), 'labels': labels,
                  'clean_predictions': torch.cat(clean_predictions),
                  'sample_ids': torch.arange(len(labels)), 'metadata': metadata}
        torch.save(cached, cache_path)
        print(f'Saved {attack_name.upper()} images: {cache_path.name}')
    return cached


def evaluate_adversarial(model, loader, criterion, attack_fn, attack_kwargs):
    name = 'fgsm' if attack_fn is fgsm_attack else 'pgd'
    cached = load_or_generate_adversarial_dataset(model, loader, criterion, name, attack_fn, attack_kwargs)
    attacked = DataLoader(TensorDataset(cached['images'], cached['labels']), batch_size=BATCH_SIZE)
    _, accuracy, _ = evaluate(model, attacked, criterion)
    correct_clean = cached['clean_predictions'].eq(cached['labels'])
    successes = 0
    offset = 0
    with torch.no_grad():
        for inputs, labels in attacked:
            wrong = model(inputs.to(DEVICE)).argmax(1).cpu().ne(labels)
            successes += (wrong & correct_clean[offset:offset + len(labels)]).sum().item()
            offset += len(labels)
    count = correct_clean.sum().item()
    asr = successes / count if count else None
    return accuracy, asr


def main():
    seed_everything(SEED)
    train_loader, val_loader, test_loader = make_loaders()
    cnn = TrafficSignCNN().to(DEVICE)
    model = PixelSpaceModel(cnn).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    if MODEL_PATH.exists():
        saved = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        # Older checkpoints contain only the CNN state dictionary.
        cnn.load_state_dict(saved['model_state_dict'] if 'model_state_dict' in saved else saved)
        print(f'Loaded checkpoint: {MODEL_PATH}')
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        best_accuracy, best_loss = -1.0, float('inf')
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(labels)
            val_loss, val_accuracy, _ = evaluate(model, val_loader, criterion)
            print(f'Epoch {epoch}/{EPOCHS}: train loss={running_loss / len(train_loader.dataset):.4f}, '
                  f'validation loss={val_loss:.4f}, validation accuracy={val_accuracy:.4f}')
            if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
                best_accuracy, best_loss = val_accuracy, val_loss
                torch.save({'model_state_dict': cnn.state_dict(), 'epoch': epoch,
                            'validation_accuracy': val_accuracy, 'validation_loss': val_loss,
                            'seed': SEED, 'preprocessing': PREPROCESSING}, MODEL_PATH)
        saved = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        cnn.load_state_dict(saved['model_state_dict'])
        print(f'Restored best validation checkpoint from epoch {saved["epoch"]}')

    loss, accuracy, macro_f1 = evaluate(model, test_loader, criterion)
    print(f'Clean test loss={loss:.4f}, accuracy={accuracy:.4f}, macro F1={macro_f1:.4f}')
    rows = []
    for epsilon in EPSILONS:
        for name, attack, kwargs in (
            ('FGSM', fgsm_attack, {'epsilon': epsilon}),
            ('PGD', pgd_attack, {'epsilon': epsilon, 'alpha': 0.005, 'iters': 20, 'restarts': 2}),
        ):
            adv_accuracy, asr = evaluate_adversarial(model, test_loader, criterion, attack, kwargs)
            if epsilon == 0 and abs(adv_accuracy - accuracy) > 1e-12:
                raise ValueError('Epsilon-zero accuracy must match clean accuracy')
            rows.append({'attack': name, **kwargs, 'adversarial_accuracy': adv_accuracy,
                         'attack_success_rate': asr, 'clean_accuracy': accuracy,
                         'accuracy_change': adv_accuracy - accuracy})
            print(f'{name} epsilon={epsilon}: accuracy={adv_accuracy:.4f}, ASR={asr}')
    (ROOT / 'baseline_results.json').write_text(json.dumps(
        {'seed': SEED, 'model_hash': model_fingerprint(model), 'preprocessing': PREPROCESSING,
         'clean_macro_f1': macro_f1, 'results': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
