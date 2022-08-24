from functools import reduce

def process_func(arr):
    new_arr = []
    if len(arr) == 1:
        new_arr = arr
    else:
        for i, _ in enumerate(arr):
            new_arr.append(reduce(lambda x, y: x*y, arr[:i]+arr[i+1:]))

    return new_arr

if __name__ == "__main__":
    input_array = [1, 2, 3]
    print(process_func(input_array))
