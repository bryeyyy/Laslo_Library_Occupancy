import random

def randgen():
    iteration = random.randint(1,100)
    return iteration

def boollist():
    checker = False
    bools = []
    for i in range(5):
        sample = randgen()
        if sample%2 == 0:
            checker = True
        else:
            checker = False
        print(sample)
        bools.append(checker)
    print(bools)

boollist()
boollist()
boollist()
boollist()
