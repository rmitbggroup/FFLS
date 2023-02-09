def set_bit(x, pos):
    """
    First create a mask which has 1 in the required position and everywhere else is 0
    Then do bitwise OR that will the set 1 in the position
    """
    mask = 1 << pos
    return x | mask


def clear_bit(x, pos):
    """
    First create a mask which has 0 in the required position and everywhere else is 1
    Same logic as set_bit then use NOT operation
    Then do bitwise AND that will the set 0 in the position
    """
    mask = 1 << pos
    return x & ~mask


def get_bit(x, pos):
    """
    First shift the required bit to right most position
    Perform bitwise AND operation with 1 to get the right most bit
    """
    return (x >> pos) & 1


def flip_bit(x, pos):
    """
    On flipping a bit we need an operator which can perform the following operation
    X - 1
    Mask - 1
    Result - 0
    X - 0
    Mask - 1
    Result - 1
    This is exactly what XOR does
    """
    mask = 1 << pos
    return x ^ mask


def is_bit_set(x, pos):
    """
    First get the bit using get_bit logic, then compare with 1
    """
    return (x >> pos) & 1 == 1


def modify_bit(x, pos, op):
    """
    Perform operation according to input op
    """
    if op == 1:
        set_bit(x, pos)
    else:
        clear_bit(x, pos)


def is_even(x):
    """
    Logic is all the even numbers in binary end with 0
    """
    return x & 1 == 0


def is_odd(x):
    """
    Logic is all the odd numbers in binary end with 1
    """
    return x & 1 == 1


def is_multiple_of_2(x):
    """
    If a binary number is a multiple of 2, only one bit will be set
    So if we substract 1 from it, the number will be a compliment
    Then do bitwise AND should yield 0
    """
    return (x & x - 1) == 0


def count_set_bits(num):
    """
    Find the number of set bits in an input integer
    Set bits are bits that set to 1
    It is also called hamming distance
    """
    count = 0
    while num:
        count += num & 1
        num >>= 1
    return count


def count_set_bits_brian_kernighan(num):
    """
    Find the number of set bits in an input integer
    Set bits are bits that set to 1
    It is also called hamming distance
    Brian Kernighan’s Algorithm
    Subtract a number by 1
    Do bitwise & with itself (n & (n-1)),
    It unset the rightmost set bit.
    """
    count = 0
    while num:
        count += 1
        num &= num - 1
    return count