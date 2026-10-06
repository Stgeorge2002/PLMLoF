"""ESM2 unique-sequence embedding used by the precompute script."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from plmlof.sites import token_index

logger = logging.getLogger(__name__)


class _LenSortedSeqs(Dataset):
    def __init__(self, sequences: list[str]):
        self.sequences = sorted(sequences, key=len)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


class SiteBank:
    """Packed residue tokens: one ``[N, D]`` store plus an int index.

    A ``dict[(seq, residue)] → Tensor[D]`` OOMs on the ESM2-650M unique-seq
    pass — millions of PyTorch objects plus a full ``[B, T, D]`` float32 copy
    every batch. This keeps one array (RAM or a scratch memmap) and never
    materialises the unused tokens.
    """

    def __init__(self, n_slots: int, dim: int, path: Path | None = None):
        if n_slots < 0 or dim < 1:
            raise ValueError(f"n_slots={n_slots} dim={dim}")
        self.dim = int(dim)
        self._index: dict[tuple[str, int], int] = {}
        self._next = 0
        self._path = Path(path) if path is not None else None
        shape = (max(int(n_slots), 1), self.dim)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if self._path.exists():
                self._path.unlink()
            self._store = np.memmap(self._path, dtype=np.float32, mode="w+", shape=shape)
        else:
            self._store = np.zeros(shape, dtype=np.float32)

    def add(self, seq: str, residues: list[int], vecs: torch.Tensor) -> None:
        if vecs.ndim != 2 or vecs.size(0) != len(residues) or vecs.size(1) != self.dim:
            raise ValueError(
                f"vecs {tuple(vecs.shape)} does not match residues={len(residues)} D={self.dim}"
            )
        n = len(residues)
        start = self._next
        if start + n > self._store.shape[0]:
            raise RuntimeError(
                f"site bank overflow: need {start + n} slots, have {self._store.shape[0]}"
            )
        self._store[start:start + n] = vecs.detach().float().cpu().numpy()
        for i, residue in enumerate(residues):
            self._index[(seq, int(residue))] = start + i
        self._next = start + n

    def get(self, seq: str, residue: int) -> np.ndarray | None:
        row = self._index.get((seq, int(residue)))
        if row is None:
            return None
        return np.asarray(self._store[row])

    def close(self) -> None:
        store = getattr(self, "_store", None)
        if store is not None and hasattr(store, "flush"):
            store.flush()
        self._store = None
        self._index.clear()
        if self._path is not None and self._path.exists():
            self._path.unlink()
            self._path = None


def _pool(hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask_f = mask.unsqueeze(-1).float()
    mean_p = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
    masked = hidden.masked_fill(~mask.unsqueeze(-1).bool(), float("-inf"))
    max_p = masked.max(dim=1).values
    max_p = max_p.masked_fill(max_p == float("-inf"), 0.0)
    return mean_p, max_p


def _hidden_dim(model) -> int:
    cfg = getattr(model, "config", None)
    dim = getattr(cfg, "hidden_size", None) if cfg is not None else None
    if dim:
        return int(dim)
    raise ValueError("model.config.hidden_size is required to bank residue tokens")


def forward_hidden_and_logits(model, ids: torch.Tensor, mask: torch.Tensor):
    """Hidden tokens plus MLM logits.

    ``EsmForMaskedLM`` returns ``MaskedLMOutput`` (logits only). Taking
    ``output_hidden_states=True`` would stash every layer and OOM. The trunk
    ``model.esm`` still exposes ``last_hidden_state``; ``lm_head`` maps it to V.
    """
    if hasattr(model, "esm"):
        hidden = model.esm(ids, attention_mask=mask).last_hidden_state
        logits = model.lm_head(hidden) if hasattr(model, "lm_head") else None
        return hidden, logits
    out = model(ids, attention_mask=mask)
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is None:
        raise AttributeError(
            f"{type(out).__name__} has no last_hidden_state; expected EsmModel or EsmForMaskedLM"
        )
    return hidden, getattr(out, "logits", None)


@torch.no_grad()
def embed_unique_sequences(
    sequences: list[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
    num_workers: int = 4,
    residue_requests: dict[str, set[int]] | None = None,
    site_store_path: Path | str | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor, SiteBank | None, SiteBank | None]:
    """Pool unique sequences; optionally bank residue tokens and MLM logits."""
    ds = _LenSortedSeqs(sequences)

    def collate(batch: list[str]) -> dict:
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        )
        enc["sequences"] = batch
        return enc

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    means, maxes, ordered = [], [], []
    site_bank: SiteBank | None = None
    logit_bank: SiteBank | None = None
    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    has_lm = hasattr(model, "lm_head") and vocab > 0
    if residue_requests:
        n_slots = sum(len(v) for v in residue_requests.values())
        path = Path(site_store_path) if site_store_path is not None else None
        site_bank = SiteBank(n_slots, _hidden_dim(model), path)
        logger.info(
            "Site bank  slots=%s  D=%s  store=%s",
            n_slots, site_bank.dim, path if path is not None else "ram",
        )
        if has_lm:
            logit_path = path.with_name(path.name + ".logits") if path is not None else None
            logit_bank = SiteBank(n_slots, vocab, logit_path)
            logger.info("Logit bank  slots=%s  V=%s", n_slots, vocab)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16
    if use_amp:
        major, _ = torch.cuda.get_device_capability(device)
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16

    for batch in tqdm(loader, desc="ESM2 unique sequences"):
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            hidden, logits = forward_hidden_and_logits(model, ids, mask)
        mean_p, max_p = _pool(hidden, mask)
        means.append(mean_p.float().cpu())
        maxes.append(max_p.float().cpu())
        seqs = batch["sequences"]
        ordered.extend(seqs)
        if site_bank is not None and residue_requests:
            n_tok = hidden.size(1)
            for b, seq in enumerate(seqs):
                requested = residue_requests.get(seq)
                if not requested:
                    continue
                residues: list[int] = []
                toks: list[int] = []
                for residue in requested:
                    tok = token_index(int(residue))
                    if 0 <= tok < n_tok:
                        residues.append(int(residue))
                        toks.append(tok)
                if not toks:
                    continue
                tok_t = torch.tensor(toks, device=hidden.device, dtype=torch.long)
                site_bank.add(seq, residues, hidden[b].index_select(0, tok_t))
                if logit_bank is not None and logits is not None:
                    logit_bank.add(seq, residues, logits[b].index_select(0, tok_t))
        del hidden
        del logits

    return ordered, torch.cat(means), torch.cat(maxes), site_bank, logit_bank
