from framework.cpp_backend import cross_entropy_loss

class CrossEntropyLoss:
    def __call__(self, logits, target):
        return cross_entropy_loss(logits, target)