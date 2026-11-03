

from .sslvmb import sslvmb

model_dict = {
    "sslvmb": sslvmb
}

def create_net(args):
    net = None
    net = model_dict[args.arch.lower()](args)
    return net