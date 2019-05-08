import torch

def _mm(t1,t2):
    if t1 is None or t2 is None:
        return None
    return torch.mm(t1,t2)
def _bmm(t1,t2):
    if t1 is None or t2 is None:
        return None
    return torch.bmm(t1,t2)

def _kron_prod(t1,t2):
    #regardless of real or imaginative
    if t1 is None or t2 is None:
        return None
    if len(t1.data.size()) == 3:
        b, n1, d1 = t1.data.size()
        is_batch = 1
    else:
        n1, d1 = t1.data.size()
        is_batch = 0
    if len(t2.data.size()) == 3:
        #batch with batch
        b2, n2, d2 = t2.data.size()
        assert(len(t1.data.size()) == 3 and b == b2)
        t1 = t1.unsqueeze(3).unsqueeze(4)
        ret = t1.repeat(1,1,n2,d2,1).view(b,n1*n2,-1)*t2.repeat(1,n1,d1)
    else:
        n2, d2 = t2.data.size()
        t1 = t1.unsqueeze(2+is_batch).unsqueeze(3+is_batch)
        if is_batch == 1:
            ret = t1.repeat(1,1,n2,d2,1).view(b, n1 * n2,-1) * t2.repeat(b, n1, d1)
        else:
            ret = t1.repeat(1,n2,d2,1).view(n1 * n2,-1) *  t2.repeat(n1, d1)
    return ret

def test_krod_prod():
    m1 = torch.tensor([[1,2],[3,4]])
    m2 = torch.tensor([[0,5],[6,7]])
    print(_kron_prod(m1,m2))
    print(_kron_prod(m1.repeat(3,1,1),m2))
    print(_kron_prod(m1.repeat(3,1,1),m2.repeat(3,1,1)))
    return None
    

def _complex_template(t1r, t1i, t2r, t2i, func):
    #func must return None if either argument is None
    ret_real_1 = func(t1r, t2r)
    #TODO: testing reverting it back to real, remove this when done
    #return ret_real_1, None

    ret_real_2 = func(t1i, t2i)
    #print(t1r.data.size(), t1i.data.size(), t2r.data.size(), t2i.data.size())
    ret_real = ret_real_1 if ret_real_2 is None else ret_real_1 - ret_real_2

    ret_imag_1 = func(t1r, t2i)
    #print("ret_imag_1", ret_imag_1)
    ret_imag_2 = func(t1i, t2r)
    if ret_imag_1 is None:
        ret_imag = None if ret_imag_2 is None else ret_imag_2
    else:
        ret_imag = ret_imag_1 if ret_imag_2 is None else ret_imag_1 + ret_imag_2
    return ret_real, ret_imag


def C_kron_prod(t1r, t1i, t2r, t2i):
    return _complex_template(t1r, t1i, t2r, t2i, _kron_prod)

def C_mm(t1r, t1i, t2r, t2i):
    #print(t1r, t1i, t2r, t2i)
    return _complex_template(t1r, t1i, t2r, t2i, _mm)

def C_bmm(t1r, t1i, t2r, t2i):
    return _complex_template(t1r, t1i, t2r, t2i, _bmm)


#test_krod_prod()