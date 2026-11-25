from src.env.binary_tree_env_one_step_likelihood import PhylogenticTreeEnv
from src.env.phylo_network_env import PhyloNetworkEnv


def build_env(cfg, all_seqs):
    assert cfg.ENV.ENVIRONMENT_TYPE in ['ONE_STEP_BINARY_TREE', 'ONE_STEP_LEVEL1_NETWORK']
    if cfg.ENV.ENVIRONMENT_TYPE == 'ONE_STEP_LEVEL1_NETWORK':
        # PhyloGFN-Net: level-1 phylogenetic networks (MERGE / RETICULATE actions)
        env = PhyloNetworkEnv(cfg, all_seqs)
    else:
        env = PhylogenticTreeEnv(cfg, all_seqs)

    return env
