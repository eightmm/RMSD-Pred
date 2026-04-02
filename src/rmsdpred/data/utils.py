import pickle


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def save_obj(data, name):
    with open(f'{name}.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def is_one(x, allowable_set):
    return [1 if x in allowable_set else 0]
