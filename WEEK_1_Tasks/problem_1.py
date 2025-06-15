import numpy as np
arr=np.random.randint(1,51, size=(5,4))
print(arr)
print("Anti-diagnol elements:")
i=0
arr1=[]
while i<4:
    arr1.append(arr[i][3-i])
    i=i+1
arr1=np.array(arr1)
print(arr1)
arr2=np.max(arr, axis=1)
print(arr2)
arr3=arr[arr<=arr.mean()]
print(arr3)

def numpy_boundary_traversal(matrix):
    boundary=[]
    boundary+=list(matrix[0, :-1])
    boundary+=list(matrix[:-1,-1])
    boundary+=list(matrix[-1,-1::-1])
    boundary+=list(matrix[-2:0:-1,0])

    return boundary
arr4=numpy_boundary_traversal(arr)
arr4=np.array(arr4)
print(arr4)