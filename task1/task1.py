from functools import reduce

def process_func(arr):
    new_arr=list(
        map(lambda indx: reduce(lambda x, y: x*y, arr[:indx]+arr[indx+1:]) if len(arr)>1 else arr[0], range(len(arr)))
    )

    return new_arr

if __name__ == "__main__":
    input_array = [1, 2, 3]

    assert len(input_array) > 0, "Согласно заданию, список долен быть ненулевой длины"
    assert set([type(elem) for elem in input_array])=={int}, "Согласно заданию, список должен быть только из целочисленных элементов"

    print(process_func(input_array))
