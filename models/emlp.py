from models.model import Model

from emlp.reps import Rep
from emlp.nn.haiku import uniform_rep, EMLPBlock, Linear, Sequential

def EMLP(rep_in,rep_out,group,ch=384,num_layers=3):
    rep_in =rep_in(group)
    rep_out = rep_out; rep_out = rep_out(group)
    # Parse ch as a single int, a sequence of ints, a single Rep, a sequence of Reps
    if isinstance(ch,int): middle_layers = num_layers*[uniform_rep(ch,group)]
    elif isinstance(ch,Rep): middle_layers = num_layers*[ch(group)]
    else: middle_layers = [(c(group) if isinstance(c,Rep) else uniform_rep(c,group)) for c in ch]
    # assert all((not rep.G is None) for rep in middle_layers[0].reps)
    reps = [rep_in]+middle_layers
    # logging.info(f"Reps: {reps}")
    def emlp(batch, key):
        mlp = Sequential(
            *[EMLPBlock(rin,rout) for rin,rout in zip(reps,reps[1:])],
            Linear(reps[-1],rep_out)
        )
        return mlp(batch.x)
    
    model = Model(model=emlp)
    return model