import numpy as np

def dot_product(u,v):
    if u.shape != v.shape:
        return "They are not of the same shape!!!"

    result = 0
    size = u.shape[0]
    if len(u.shape) == 1:
        for i in range(size):
            result += u[i] * v[i]
    else:
        for i in range(size):
            result += dot_product(u[i],v[i])

    return result


def cross_product (u,v):
    num_row_u = u.shape[0]
    num_row_v = v.shape[0]
    try:
        num_col_u = u.shape[1]
    except:
        num_col_u = 1
    try:
        num_col_v = v.shape[1]
    except:
        num_col_v = 1

    result = np.zeros((num_row_u,num_col_v))

    if num_col_u == num_row_v:

        if num_col_v == 1:
            result = np.zeros(num_row_u)
            for row in range(num_row_u):
                result[row] = dot_product(u[row],v)
        else:
            for row in range(num_row_u):
                for col in range(num_col_v):
                    result[row][col] = dot_product(u[row],v[:,col])
        
        return result
    
    else:
        return print("The vectors can't be multiplied because: ", num_col_u , ' isnot equal to' , num_row_v)