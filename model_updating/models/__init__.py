from .cvae import CVAE
from .cgan import CGAN
from .cnf import CNF
from .cddpm import CDDPM
from .cfm import CFM

REGISTRY = {
    "cvae": CVAE,
    "cgan": CGAN,
    "cnf": CNF,
    "cddpm": CDDPM,
    "cfm": CFM,
}
