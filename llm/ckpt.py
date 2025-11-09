import os
import typing
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    # should dump all the state from the
    # first three parameters into the file-like object out. You can use the state_dict method of both
    # the model and the optimizer to get their relevant states and use torch.save(obj, out) to dump
    # obj into out (PyTorch supports either a path or a file-like object here). A typical choice is to
    # have obj be a dictionary, but you can use whatever format you want as long as you can load your
    # checkpoint later.
    # This function expects the following parameters:
    # model: torch.nn.Module
    # optimizer: torch.optim.Optimizer
    # iteration: int
    # out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    # should load a checkpoint from src (path or file-
    # like object), and then recover the model and optimizer states from that checkpoint. Your
    # function should return the iteration number that was saved to the checkpoint. You can use
    # torch.load(src) to recover what you saved in your save_checkpoint implementation, and the
    # load_state_dict method in both the model and optimizers to return them to their previous
    # states.
    # This function expects the following parameters:
    # src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    # model: torch.nn.Module
    # optimizer: torch.optim.Optimizer
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
