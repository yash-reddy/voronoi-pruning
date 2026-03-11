"""Utility functions for Voronoi Token Pruning."""

import argparse
import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

MAX_SCORE = int(1e8)
MAX_BATCH_SIZE = 5000  # Depends on the VRAM, this should work for VRAM >=32G
BELOW_MIN_SCORE = -2.0  # score assigned to masked tokens, should be lower than any valid score.


# logger is explicitly passed to ensure that this function can be used by other routines as well.
def log_args(args: argparse.Namespace, logger_object: logging.Logger) -> None:
    """Logs the arguments used to run the script."""
    logger_object.info("Running the script with the following Arguments:")
    for arg, value in vars(args).items():
        logger_object.info("%20s: %s", arg, value)
    logger_object.info("******** End of Arguments ********")


def sample_in_unit_ball(
    n: int,
    num_points: int = 1,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    sample_on_surface: bool = False,
) -> torch.Tensor:
    """Generates random samples uniformly distributed inside an n-dimensional unit ball."""
    dir_vectors = torch.randn(num_points, n, device=device, dtype=dtype)
    dir_vectors = dir_vectors / dir_vectors.norm(dim=1, keepdim=True)
    if not sample_on_surface:
        norm_samples = torch.rand(num_points, device=device, dtype=dtype) ** (1 / n)
        dir_vectors = dir_vectors * norm_samples.unsqueeze(1)

    return dir_vectors


def get_prune_targets(
    padded_matrix: torch.Tensor,
    vvp_mask: torch.Tensor,
    sampled_points: torch.Tensor,
    step_size: int = 1,
    iterative: bool = True,
    batch_size: int = 100,
    beam_size: int = 1,
    use_relu: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Main function to get pruning targets and scores using the Voronoi Pruning method."""
    if beam_size > 1:
        assert iterative, "Beam Search can only be used in iterative mode"
        assert step_size == 1, "Beam Search can only be used with step_size=1"

        return beam_search_prune_targets(
            padded_matrix,
            vvp_mask,
            sampled_points,
            batch_size=batch_size,
            beam_size=beam_size,
            use_relu=use_relu,
        )
    else:
        return get_iterative_prune_targets(
            padded_matrix,
            vvp_mask,
            sampled_points,
            step_size=step_size,
            batch_size=batch_size,
            use_relu=use_relu,
        )


@torch.no_grad()
def beam_search_prune_targets(
    padded_matrix: torch.Tensor,
    vvp_mask: torch.Tensor,
    sampled_points: torch.Tensor,
    batch_size: int = 100,
    beam_size: int = 3,
    use_relu: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Beam search approach for token pruning."""
    device = padded_matrix.device
    n_docs, max_doclen = vvp_mask.shape

    total_capacity = beam_size * n_docs * max_doclen
    prune_targets = torch.empty((total_capacity, 2), dtype=torch.int32, device=device)
    prune_scores = torch.empty((total_capacity,), dtype=torch.float32, device=device)
    ptr = 0

    for start in range(0, n_docs, batch_size):
        end = min(start + batch_size, n_docs)
        bat_size = end - start

        batch = padded_matrix[start:end]  # (B, L, D)
        base_mask = vvp_mask[start:end]  # (B, L)
        batch = batch.masked_fill(base_mask.unsqueeze(-1), -float("inf"))

        dp = batch @ sampled_points.T  # (B, L, S)
        if use_relu:
            dp = dp.relu()
        dp = torch.nan_to_num(dp, nan=BELOW_MIN_SCORE)

        # ---- initialize beam state ----
        mask = base_mask.unsqueeze(1).expand(bat_size, beam_size, max_doclen).clone()
        cum_error = torch.zeros(bat_size, beam_size, device=device)
        history_tokens = torch.full((bat_size, beam_size, 0), -1, dtype=torch.int32, device=device)
        history_errors = torch.zeros((bat_size, beam_size, 0), dtype=torch.float32, device=device)

        # ---- First Step ----
        # compute margin-based step errors
        top1, top1_idx = dp.max(dim=1)  # (B, S)
        dp_temp = dp.clone()
        dp_temp.scatter_(1, top1_idx.unsqueeze(1), BELOW_MIN_SCORE)
        top2, _ = dp_temp.max(dim=1)
        margins = (top1 - top2).float()

        step_error = torch.zeros(bat_size, max_doclen, device=device, dtype=margins.dtype)
        step_error.scatter_add_(1, top1_idx, margins)  # sum contributions
        step_error.masked_fill_(base_mask, MAX_SCORE)

        # pick top beam_size tokens per doc
        topk_scores, topk_tokens = torch.topk(step_error, k=beam_size, largest=False, dim=1)

        # append to history
        history_tokens = torch.cat(
            [history_tokens, topk_tokens.unsqueeze(-1)], dim=-1
        )  # (B, beam_size, 1)
        history_errors = torch.cat([history_errors, topk_scores.unsqueeze(-1)], dim=-1)
        cum_error = topk_scores.clone()

        # mark the first tokens in mask
        for b in range(bat_size):
            for beam_idx in range(beam_size):
                mask[b, beam_idx, topk_tokens[b, beam_idx]] = True

        # ---- Remaining Steps ----
        for _ in range(1, max_doclen):
            dp_beam = dp.unsqueeze(1).expand(-1, beam_size, -1, -1).clone()
            dp_beam = dp_beam.masked_fill(mask.unsqueeze(-1), BELOW_MIN_SCORE)

            top1, top1_idx = dp_beam.max(dim=2)
            dp_beam.scatter_(2, top1_idx.unsqueeze(2), BELOW_MIN_SCORE)
            top2, _ = dp_beam.max(dim=2)
            margins = (top1 - top2).float()  # (B, beam_size, S)

            step_error = torch.zeros(
                bat_size, beam_size, max_doclen, device=device, dtype=margins.dtype
            )
            step_error.scatter_add_(2, top1_idx, margins)
            step_error.masked_fill_(mask, MAX_SCORE)

            total_error = cum_error.unsqueeze(-1) + step_error
            flat_error = total_error.view(bat_size, -1)
            _, idxs = torch.topk(flat_error, k=beam_size, largest=False)
            beam_ids = idxs // max_doclen
            token_ids = idxs % max_doclen

            mask = mask.gather(1, beam_ids.unsqueeze(-1).expand(-1, -1, max_doclen))
            cum_error = cum_error.gather(1, beam_ids)
            history_tokens = history_tokens.gather(
                1, beam_ids.unsqueeze(-1).expand(-1, -1, history_tokens.shape[-1])
            )
            history_errors = history_errors.gather(
                1, beam_ids.unsqueeze(-1).expand(-1, -1, history_errors.shape[-1])
            )

            selected_step_error = (
                step_error.gather(1, beam_ids.unsqueeze(-1).expand(-1, -1, max_doclen))
                .gather(2, token_ids.unsqueeze(-1))
                .squeeze(-1)
            )
            mask.scatter_(2, token_ids.unsqueeze(-1), True)
            cum_error += selected_step_error
            history_tokens = torch.cat(
                [history_tokens, token_ids.unsqueeze(-1).to(torch.int32)], dim=-1
            )
            history_errors = torch.cat([history_errors, selected_step_error.unsqueeze(-1)], dim=-1)

        # ---- Extract best beam per document ----
        valid_counts = (~base_mask).sum(dim=1)  # (B,)

        # compute true cumulative error using only valid steps
        true_total_error = torch.zeros(bat_size, beam_size, device=device)

        for b in range(bat_size):
            doclen = valid_counts[b]
            true_total_error[b] = history_errors[b, :, :doclen].sum(dim=-1)

        best_beam_idx = true_total_error.argmin(dim=1)

        for b in range(bat_size):
            doclen = valid_counts[b]
            best_tokens = history_tokens[b, best_beam_idx[b], :doclen]
            best_errors = history_errors[b, best_beam_idx[b], :doclen]

            for t in range(doclen):
                prune_targets[ptr, 0] = start + b
                prune_targets[ptr, 1] = best_tokens[t]
                prune_scores[ptr] = best_errors[t]
                ptr += 1

    return prune_targets[:ptr], prune_scores[:ptr]


@torch.no_grad()
def get_iterative_prune_targets(
    padded_matrix: torch.Tensor,
    vvp_mask: torch.Tensor,
    sampled_points: torch.Tensor,
    step_size: int = 1,
    batch_size: int = 100,
    use_relu: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs fast iterative pruning by computing mean error scores and selecting lowest scoring tokens."""
    n_docs, max_doclen = vvp_mask.shape
    if step_size > max_doclen:
        step_size = max_doclen
        logger.warning(
            "step_size (%d) exceeds max_doclen (%d), clamping step_size to max_doclen",
            step_size,
            max_doclen,
        )

    num_steps = (max_doclen + step_size - 1) // step_size
    total_prune_targets = num_steps * n_docs * step_size

    device = padded_matrix.device
    prune_targets = torch.empty((total_prune_targets, 2), dtype=torch.int32, device=device)
    prune_scores = torch.empty((total_prune_targets,), dtype=torch.float32, device=device)
    targets_pointer = 0

    for start in range(0, n_docs, batch_size):
        end = min(start + batch_size, n_docs)
        batch = padded_matrix[start:end]  # (B, L, N)
        bat_mask = vvp_mask[start:end].clone()  # (B, L)
        bat_size = batch.shape[0]
        batch[bat_mask] = float("-inf")
        dot_products = torch.matmul(batch, sampled_points.T)  # (B, L, S)
        if use_relu:
            dot_products = dot_products.relu()
        dot_products = torch.nan_to_num(dot_products, nan=BELOW_MIN_SCORE)
        # The only reason padded tokens will be considered is if there's just one valid token remaining.
        for _ in range(num_steps):
            top1_vals, top1_idx = dot_products.max(dim=1)
            dot_products.scatter_(1, top1_idx.unsqueeze(1), BELOW_MIN_SCORE)
            top2_vals, _ = dot_products.max(dim=1)
            dot_products.scatter_(1, top1_idx.unsqueeze(1), top1_vals.unsqueeze(1))

            margins = top1_vals - top2_vals
            winners = top1_idx
            total_errors = torch.zeros((bat_size, max_doclen), dtype=torch.float32, device=device)
            total_errors.scatter_add_(dim=1, index=winners, src=margins.float())
            total_errors[bat_mask] = MAX_SCORE
            min_scores, e_idxs = torch.topk(total_errors, k=step_size, dim=1, largest=False)
            batch_indices = (
                torch.arange(start, end, device=device).unsqueeze(1).expand(-1, step_size)
            )
            flat_size = bat_size * step_size
            prune_targets[targets_pointer : targets_pointer + flat_size, 0] = batch_indices.reshape(
                -1
            )
            prune_targets[targets_pointer : targets_pointer + flat_size, 1] = e_idxs.reshape(-1)
            prune_scores[targets_pointer : targets_pointer + flat_size] = min_scores.reshape(-1)
            targets_pointer += flat_size
            # Update mask with newly pruned indices
            b_idx = torch.arange(bat_size, device=device).unsqueeze(1)  # (B, 1)
            bat_mask[b_idx, e_idxs] = True  # boolean mask update
            # Only mask the selected tokens, not the entire 3D tensor
            dot_products[b_idx, e_idxs] = BELOW_MIN_SCORE
    return (
        prune_targets[:targets_pointer],
        prune_scores[:targets_pointer],
    )
