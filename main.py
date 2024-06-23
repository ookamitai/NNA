from random import randint, uniform
import numpy as np

class TrainingData:
    size: int = None # number of example data
    data: dict[tuple[int, int], int] = {}# stores the example data as dict

    def __init__(self, size: int) -> None:
        self.size = size

    def init_data(self) -> None:
        for i in range(self.size):
            num1 = randint(0, 100)
            num2 = randint(0, 100)
            self.data[(num1, num2)] = num1 + num2
        
class Network:
    def __init__(self, data: dict[tuple[int, int], int]) -> None:
        self.x1 = uniform(-1,1) 
        self.x2 = uniform(-1,1) 
        self.y1 = uniform(-1,1) 
        self.y2 = uniform(-1,1) 
        self.z1 = uniform(-1,1) 
        self.z2 = uniform(-1,1) 
        self.data = data

    @staticmethod
    def sigmoid(x: float) -> float:
        # this is the activation method
        return 1 / (1 + np.exp(-x))

    def infer(self, num1, num2) -> float:
        num1, num2 = num1 / 100, num2 / 100
        h1n1 = self.sigmoid(num1 * self.x1 + num2 * self.y1)
        h1n2 = self.sigmoid(num1 * self.x2 + num2 * self.y2)
        h2n1 = h1n1 * self.z1 + h1n2 * self.z2
        return h2n1 * 100
    
    @staticmethod
    def loss(value: float, target: float) -> float:
        # use mse to calc loss
        return (value - target) ** 2
    
    def train(self, lr: float, iter: int) -> None:
        for epoch in range(1, iter + 1):
            ls = []
            for key in self.data:
                num1, num2 = key
                target = self.data[key]
                # rescale
                num1, num2 = num1 / 100, num2 / 100
                target /= 100

                # calc output
                h1n1 = self.sigmoid(num1 * self.x1 + num2 * self.y1)
                h1n2 = self.sigmoid(num1 * self.x2 + num2 * self.y2)
                output = h1n1 * self.z1 + h1n2 * self.z2
                loss = self.loss(output, target)

                self.x1 -= lr * -2 * (target - output) * self.z1 * h1n1 * (1 - h1n1) * num1
                self.y1 -= lr * -2 * (target - output) * self.z1 * h1n1 * (1 - h1n1) * num2
                self.x2 -= lr * -2 * (target - output) * self.z2 * h1n2 * (1 - h1n2) * num1
                self.y2 -= lr * -2 * (target - output) * self.z2 * h1n2 * (1 - h1n2) * num2

                self.z1 -= lr * -2 * (target - output) * h1n1
                self.z2 -= lr * -2 * (target - output) * h1n2
                ls.append(loss)
            if epoch % 100 == 0:
                print(f"{epoch=}, mean loss={sum(ls) / len(ls)}")

def main(size, iter, lr):
    print(f"{size=}, {iter=}, {lr=}")
    data = TrainingData(size)
    data.init_data()

    n = Network(data.data)
    n.train(lr, iter)
    print("\n########## INFERENCE ##########")
    for _ in range(5):
        n1 = randint(0, 100)
        n2 = randint(0, 100)
        print(f"{n1} + {n2} = {n.infer(n1, n2)}")

if __name__ == "__main__":
    main(1000, 5000, 0.1)
