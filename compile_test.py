import logging

import torch
from torch import nn
import click

import natten
from natten import NeighborhoodAttention2D

#torch._logging.set_logs(dynamo=logging.DEBUG)
#torch._inductor.config.debug = True


@click.command()
@click.option("--backend", type=str, default="inductor")
# To see behavior as of main branch, don't pass to see 
# it after we register ops natively.
@click.option("--use-non-native-op", is_flag=True)
# In case we want to see behavior on standard torch modules
# and ops.
@click.option("--skip-natten-ops", is_flag=True)
@click.option("--fullgraph", is_flag=True)
def main(
    backend: str,
    use_non_native_op: bool,
    skip_natten_ops: bool,
    fullgraph: bool,
):
    # NOTE: I'm only registering the fused op as a native
    # torch op under src/natten/backend.py.
    # This means switching between the two types of ops
    # lets us observe what happens to torch.compile under
    # different versions (kind of.)
    if use_non_native_op:
        natten.use_fused_na(False)
    else:
        natten.use_fused_na(True)

    with torch.cuda.device(0):
        device = 'cuda'
        dtype = torch.float16
        B, H, W, C = 2, 56, 56, 128
        heads = 4

        assert C >= heads and C % heads == 0

        x = torch.randn((B, H, W, C), dtype=dtype, device=device)
        print(f"{x.shape=}, {x.dtype=}, {x.device=}")

        m_e = nn.Sequential(
            nn.Linear(C, C),
            nn.Identity() if skip_natten_ops else NeighborhoodAttention2D(
                dim=C,
                num_heads=heads,
                kernel_size=(7, 7),
                dilation=(1, 1)
            ),
            nn.Linear(C, C),
        ).to(dtype).to(device)

        print(f"Created module")
        print("calling torch.compile")

        m_compiled = torch.compile(m_e, fullgraph=fullgraph, backend=backend)

        print("done.")
        print("first forward pass")

        y = m_compiled(x)

        print("followup forward pass")
        y = m_compiled(x)


if __name__ == "__main__":
    main()
