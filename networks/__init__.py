

from .sslvmb import sslvmb, sslvmb_cr

model_dict = {
    "sslvmb": sslvmb,
    "sslvmb_cr": sslvmb_cr, # per-choice location model
}

def create_net(args):
    net = None
    net = model_dict[args.arch.lower()](args)
    return net