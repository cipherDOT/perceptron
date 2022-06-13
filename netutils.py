
# utility functions for the neural network

# ------------------------------ data seperation util -------------------------------- #

def data_split(data: list[list[int]], ratio: int) -> tuple[list[list[int]]]:

    # using the ratio, determines the amount of train and test data
    train_quantity = int(len(data) * ratio)
    train_data, test_data = data[:train_quantity], data[train_quantity:]
    return train_data, test_data

# --------------------------------- map value util ----------------------------------- #

def mapn(value, act_lower, act_upper, to_lower, to_upper):
    return to_lower + (to_upper - to_lower) * ((value - act_lower) / (act_upper - act_lower))

# ------------------------------------------------------------------------------------ #
