import numpy as np
arr=[]
for i in range(20):
    a=np.random.rand()*10
    arr.append(a)
arr=np.array(arr)
print(arr)
print(np.min(arr))
print(np.max(arr))
print(np.median(arr))
#i=0
#while i<arr.shape[0]:
#    if arr[i]<5:
#        arr[i]=arr[i]*arr[i]
#    i=i+1
#print(arr)
arr[arr<5]=arr[arr<5]**2
print(arr)

def numpy_alternate_sort(array):
    arr=np.sort(array)
    res=[]
    i=0
    j=arr.size-1
    while i<=j:
        res.append(arr[i])
        res.append(arr[j])
        i=i+1
        j=j-1
    if i==j:
        res.append(arr[i])
    return res
res=numpy_alternate_sort(arr)
res=np.array(res)
print(res)