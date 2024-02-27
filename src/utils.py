class Checkpointer:
    def __init__(self, save_fn: callable, alg_name: str) -> None:
        self.save_fn = save_fn
        self.alg_name = alg_name
        self.counter = 0
        self.best_acc = 0

    def update(self, acc: float) -> None:
        self.counter += 1
        self.save_fn(self.alg_name + '_{0:02d}.ct'.format(self.counter))

        if acc > self.best_acc:
            self.best_acc = acc
            self.save_fn(self.alg_name + '_final.ct'.format(self.counter))
