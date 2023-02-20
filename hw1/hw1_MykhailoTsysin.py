"""HW 1 source code"""

from typing import List

# Goals:
# __call__ function
# __gt__ function
# __iter__
# __len___

class Sequence():
    """Base class for a sequence"""
    def __init__(self, array: List[int]) -> None:
        self.array = array
        self.i = -1
    def __gt__(self, other): # implementation for ">" operation
        if len(self.array) != len(other.array):
            raise ValueError("Array lengths are different")
        return sum([int(a > b) for a, b in zip(self.array, other.array)])
    def __len__(self):  # support for len() operation
        return len(self.array)
    def __iter__(self): # support for iterator
        return SequenceIterator(self)

class SequenceIterator:
    """Helper class defining behavior for sequence iterator"""
    def __init__(self, sequence_obj):
        self.array = sequence_obj.array
        self.i = -1
    def __iter__(self):
        return self
    def __next__(self):
        self.i += 1
        if self.i < len(self.array):
            return self.array[self.i]
        raise StopIteration

class Fibonacci(Sequence):
    """Fibonacci sequence handler"""
    def __init__(self, first_value: int, second_value: int) -> None:
        super().__init__([])
        self.first_value = first_value  # Store initial values for the case when the class is
                                        # called with 0 or 1 as an in out and we loose data
                                        # in the main array
        self.second_value = second_value

    def __call__(self, length: int) -> None:
        # If next call length is smaller than previous, simply cut old array:
        if self.array and len(self.array) >= length:
            self.array = self.array[:length]
        elif length <= 2:   # if length is too small, we need to return array manually
            self.array = [self.first_value, self.second_value][:length]
        else:
            if len(self.array) < 2: # set up the Fibonacci base numbers again if necessary
                self.array = [self.first_value, self.second_value]
            init_len = len(self.array)
            for _ in range(length - init_len):
                self.array.append(self.array[-1] + self.array[-2])
        print(self.array)



class Prime(Sequence):
    """Prime number sequence handler"""
    def __init__(self) -> None:
        super().__init__([])

    def __call__(self, length: int) -> None:
        # If next call length is smaller than previous, simply cut old array:
        if self.array and len(self.array) >= length:
            self.array = self.array[:length]
        elif length == 0:
            self.array = []
        else:
            if not self.array:
                self.array = [2]
            candidate = self.array[-1] + 1
            while len(self.array) < length: # while the number of primes is less than needed
                found_division = False  # indicator variable for whether we found a prime
                                        # the candidate is divisible by
                for prime in self.array:
                    if not candidate % prime: # not a prime, found division
                        found_division = True
                        break
                if found_division:   # the number is not a prime
                    candidate += 1
                else:   # add the number to the prime list
                    self.array.append(candidate)
        print(self.array)

# Driver code
if __name__ == "__main__":
    PS = Prime()
    PS(3)
    assert PS.array == [2, 3, 5]
    PS(6)
    assert PS.array == [2, 3, 5, 7, 11, 13]
    PS(4)
    assert PS.array == [2, 3, 5, 7]
    PS(10)
    assert PS.array == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    PS(0)
    assert PS.array == []

    print()
##############################################
    FS = Fibonacci(1, 1)

    FS(3)
    assert FS.array == [1, 1, 2]
    FS(1)
    assert FS.array == [1]
    FS(10)
    assert FS.array == [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    FS(2)
    assert FS.array == [1, 1]
    FS(0)
    assert FS.array == []

    print()
##############################################
    FS = Fibonacci(first_value=1, second_value=2)
    FS(length=5) # [1, 2, 3, 5, 8]
    print(len(FS)) # 5
    print([n for n in FS]) # [1, 2, 3, 5, 8]

    FS = Fibonacci(first_value=2, second_value=2)
    FS(length=6) # [2, 2, 4, 6, 10, 16]
    print(len(FS)) # 5
    print([n for n in FS]) # [2, 2, 4, 6, 10, 16]
    print([n for n in FS]) # [2, 2, 4, 6, 10, 16], testing repeated use

    print()
##############################################
    PS = Prime()
    PS (length=8) # [2, 3, 5, 7, 11 , 13 , 17 , 19]
    print (len(PS)) # 8
    print ([n for n in PS]) # [2, 3, 5, 7, 11 , 13 , 17 , 19]
    print ([n for n in PS]) # [2, 3, 5, 7, 11 , 13 , 17 , 19], testing repeated use

    PS (length=6) # [2, 3, 5, 7, 11 , 13]
    print (len(PS)) # 8
    print ([n for n in PS]) # [2, 3, 5, 7, 11 , 13]
    print ([n for n in PS]) # [2, 3, 5, 7, 11 , 13], testing repeated use

    print()
##############################################
    FS = Fibonacci(first_value=1, second_value=2)
    FS(length=8) # [1, 2, 3, 5, 8, 13 , 21 , 34]
    PS = Prime()
    PS(length=8) # [2, 3, 5, 7, 11 , 13 , 17 , 19]
    print (FS > PS) # 2
    assert (FS > PS) == 2
    assert (PS > FS) == 5
    PS(length=5) # [2, 3, 5, 7, 11]
    print(FS > PS) # will raise an error
