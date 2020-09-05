def findMaxSubArray(A: list) -> list:
    """
    Performs the search of sub array with the biggest sum of its elements
    """
    N = len(A)
    # save initial array A from overwritting during calculations
    numbers = A[:]
    # subarrays[i] will contain max subarray where A[i] is the last element
    subarrays = {i: [A[i]] for i in range(N)}
    for i in range(1, N):
        # numbers[i] equals to the sum of elements
        # of max subarray where A[i] is the last element.
        # so if numbers[i - 1] <= 0,
        # we don't want to sum the sum of that subarray
        # with number[i], we should keep it equals to A[i]
        if numbers[i - 1] > 0:
            numbers[i] += numbers[i - 1]
            subarrays[i] = subarrays[i - 1] + subarrays[i]
    # find the index of the max element of numbers
    index_of_array_with_max_sum = numbers.index(max(numbers))
    # looking at the subarrays[index_of_array_with_max_sum]
    # we can see the wishful subarray
    return subarrays[index_of_array_with_max_sum]


if __name__ == "__main__":
    inputs = [[-2, 1, -3, 4, -1, 2, 1, -5, 4],
              [-5, -3, -2, -1],
              [4, -3, 5, -1, 1],
              [5, -2, float('inf')],
              [10, -20, 5, -4, 10]]

    outputs = [[4, -1, 2, 1],
               [-1],
               [4, -3, 5],
               [5, -2, float('inf')],
               [5, -4, 10]]

    for i in range(len(inputs)):
        assert findMaxSubArray(inputs[i]) == outputs[i], \
               f'Incorrect answer for {inputs[i]}'
