from src.gfn.tb_gfn_phylo import TBGFlowNetGenerator
from src.gfn.tb_gfn_net import TBGFlowNetNet


def build_gfn(cfg, env, device, ddp):
    assert cfg.GFN.LOSS_TYPE in ['TB', ]
    if cfg.ENV.ENVIRONMENT_TYPE == 'ONE_STEP_LEVEL1_NETWORK':
        generator = TBGFlowNetNet(cfg.GFN, env, device, ddp)     # PhyloGFN-Net
    else:
        generator = TBGFlowNetGenerator(cfg.GFN, env, device, ddp)
    return generator
