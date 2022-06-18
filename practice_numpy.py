import numpy as np

my_arr = np.arange(1_000_000)

my_list = list(range(1_000_000))

%timeit my_arr*2

%timeit my_list*2

data = np.array([[0.5, -1.3, 2.3], [0.3, -.5, -4.2]])

data * 10

data + data

data.shape

data * data

data

data.T.shape

np.matmul(data, data.T)

data.dtype

# Creating nd arrays from lists

data1= [1.6, -1.4, 2.4, 3.1]

arr1 = np.array(data1)

arr1.shape

arr1

# Creating nd arrays from nested sequences

data2 = [[1.4, 5.3, 2.4], [3.5, 6.4, 3.4]]

arr2 = np.array(data2)

arr2.shape

arr2

#2 dimensional array
arr2.ndim

arr2.shape

arr2.dtype

# Create an array of zeros or ones

np.zeros(10)

zero_dat = np.zeros((3,3,3))

np.ones(10)

one_dat = np.ones((3,3))

one_dat

# Broadcasting during addition
# Numpy fills in missing dimension of ones
zero_dat + one_dat

mt_dat = np.empty((3, 3))
one_dat
mt_dat

mt_dat + one_dat

arr1 + np.ones_like(arr1)

arr1 + 1

# Create a range in an array
np.arange(1,11)

np.full((3,3), fill_value=5.5)

np.full_like(arr1, fill_value=42)

np.eye(4)

np.identity(6)

# Specify data type during definition
arr3 = np.array([2, 3, 4], dtype=np.int32)

arr4 = arr3 + 1.2

arr3.dtype

arr4.dtype

# Changing data type by casting

arr5 = np.array([1, 2, 3, 4], dtype=np.int16)

arr5.dtype

float_arr = np.array(arr5, dtype=np.float32)

float_arr.dtype

# Truncation can occur when casting from float to integers
arr6 = np.array([2.34, 3.54, 4.4, 1, 0])

arr6.dtype

arr7 = np.array(arr6, dtype=np.int32)

arr7

arr8 = np.array(arr6, dtype=bool)

arr8

# Cast array of strings into floats
arr9 = np.array(["2.3", "42", "-0.1"], dtype=np.string_)

arr9.dtype

arr9.astype(dtype=float)

# Cast a another array's dtype on an array

int_array = np.array([1, 2, 3, 4], dtype=np.int32)
int_array.dtype

arr10 = np.array([2.5, 6.3, 5.3], dtype=np.float32)

int_array.astype(arr10.dtype)

# Shortcut definition of dtype

arr11 = np.array([2.3, 4.3, 6.9], dtype="u4")

arr11

#TODO https://wesmckinney.com/book/numpy-basics.html#ndarray_binops

# Arithmetic with NumPy arrays

arr12 = np.array([[1.3, 2.5, 6.9], [3.6, 9., 3.]])

arr12

arr12 * arr12

arr12 - arr12

1/arr12

arr12

arr13 = np.array([[2, 2, 2], [2, 2, 2]], dtype=float)

arr13.dtype

arr12 > arr13

arr12

# Basic index slicing
arr = np.arange(10)

arr[5]

arr[0:3]

arr[4:8]

len(arr)

arr[-1]

arr[4:8] = 12

arr

# Array slices are views on the original array
# The data are not copied
# Changes to the slice change the source
arr_slice = arr[5:8]

arr_slice

arr

arr_slice[1] = 42

arr

arr_slice[:] = 42

arr

# To prevent this behaviour you need to create a copy of the array

arr_slice_cp = arr[5:8].astype(float).copy()

arr_slice_cp[:] = 0.01

arr

arr_slice_cp
arr_slice_cp.dtype

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

arr2d[2]

np.arange(30,40)

arr2d

arr2d.shape

arr2d[2]

arr2d[2][0]

arr2d[2, 0]

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

arr3d[0]

arr3d

arr3d.ndim

arr3d.shape

len(arr3d.shape)

arr3d[1,1,0]

arr.shape

arr.squeeze().shape

arr1.shape

arr_n = np.array([[[1,2], [2,3]], [[0, 2], [2,4]], [[0, 2], [2,4]]])

arr_n.shape
# Predict what you will get
type(arr_n[1,1,0:1].squeeze())

pick = arr_n[1,1,0:1].squeeze()

pick.ndim

pick

unsqueezed = arr_n[1,1,0:1]

unsqueezed.shape

unsqueezed

old_values = arr3d[0].copy()

old_values

arr3d[0] = 42

arr3d

arr3d[1,0]

arr3d[0] = old_values

arr3d

arr3d[1,0]

x = arr3d[1]

x

# arr3d[1,0] is equivalent to x[0]
x[0]

arr[1:6]

arr2d

arr2d[2]

arr2d.shape

arr2d[0][2]

arr2d[0, 2]

# Select the first 2 rows of arr2d (along axis 0)
arr2d[:2]

# Select the first 2 rows and the second and third columns
arr2d[:2, 1:]

# Lower dimensional slice
# First row and first two columns
arr2d[0, :2]

# Second row and first two columns
lower_dimensional_slice = arr2d[1, :2]
lower_dimensional_slice.shape
lower_dimensional_slice

# Select the 3rd column and the first 2 rows
arr2d[:2, 2]
arr2d

# Select second column and all rows
arr2d[:, 1]

# A colon by itself means to take the entire axis
arr2d[:, :1]

# Assigning to a slice expression assigns to the whole selection
arr2d[:2, :1]

old_num = arr2d[:2, 1:]

arr2d[:2, 1:] = 0

arr2d

# Boolean indexing
names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])

data = np.array([[4, 7], [0, 2], [-5, 6], [0, 0], [1, 2],
                 [-12, -4], [3, 4]])

names
data

names == "Bob"

data[names == "Bob"]

# Select the first element of values matching Bob 
data[names == "Bob", 0]

data[names == "Bob", 1:].shape

data[names == "Bob", 1].shape

# To select everything but "Bob", you can either use != or negate the condition using ~

~(names == "Bob")
data[~(names == "Bob")]
data[names != "Bob"]

# Using masks to select 2 of the 3 names
mask = (names == "Bob") | (names == "Will")

mask

data[mask]

# Booleans in numpy use & and | not 'and' and 'or'

data
# Set all the negative values to 0
data[data < 0] = 0

data

data[names != "Joe"] = 7

data

# Fancy Indexing 
# https://wesmckinney.com/book/numpy-basics.html#ndarray_fancy
import numpy as np
arr = np.zeros((8, 4))

arr

for i in range(8):
    arr[i] = i
    
arr

arr[[4, 3, 0, 6]]

arr[[4, 4, 4]]

arr[[-1, -2, -4]]

arr = np.arange(32).reshape((8, 4))

arr

# selects a one-dimensional array of elements corresponding to each tuple of indices
arr[[1, 5, 7, 2], [0, 3, 1, 2]]

arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]

# If you assign values with fancy indexing, the indexed values will be modified:
arr
arr[[1, 5, 7, 2], [0, 3, 1, 2]]

arr[[1, 5, 7, 2], [0, 3, 1, 2]] = 0

arr

# Transposing arrays
arr.T


arr = np.array([[0, 1, 0], [1, 2, -2], [6, 3, 2], [-1, 0, -1], [1, 0, 1]])

arr.T
arr


np.dot(arr.T, arr)


sum(arr.T[0,:] * arr[:,0])

sum(arr.T[1,:]*arr[:,0])

arr.T.shape[0]

# Manual calculation of dot product
dot_prod_list = [sum(arr.T[i,:]*arr[:,j]) 
 for i in range(arr.T.shape[0]) 
 for j in range(arr.T.shape[0])]
dot_prod = np.array(dot_prod_list).reshape((3,3))

dot_prod.shape

dot_prod

# Using the infix operator
arr.T @ arr


# Swap axes
arr
np.swapaxes(arr, 1,0)
np.swapaxes(arr,0,1)

# Pseudorandom number generation
samples = np.random.standard_normal(size=(3,3))
sample_uniform = np.random.uniform(size=(3,3)) - 0.5
sample_uniform

# https://wesmckinney.com/book/numpy-basics.html#numpy_random







