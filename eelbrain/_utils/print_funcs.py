# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"Some functions which print data in a different way"
import numpy as np


def dicttree(dictionary):
    """Print a hierarchical dictionary"""
    print('\n'.join(_dict_repr(dictionary)))


def _dict_repr(dictionary, indent=0):
    out = []
    head = ' ' * indent
    for k, v in dictionary.items():
        if isinstance(v, dict):
            out.append(head + repr(k))
            out.extend(_dict_repr(v, indent=indent + 3))
        else:
            r = repr_1line(v, w=40)
            out.append(head + repr(k) + ': ' + r)
    return out


def printdict(dictionary, w=100, fmt='%r', sort=True, max_v_lines=6):
    """Print only one key-value pair per line"""
    print(strdict(dictionary, w=w, fmt=fmt, sort=sort, max_v_lines=max_v_lines))


def strdict(dictionary, w=100, fmt='%r', sort=True, max_v_lines=6):
    items = []
    k_len = 0
    keys = sorted(dictionary) if sort else dictionary.keys()
    for k in keys:
        v = dictionary[k]
        k = str(k) if isinstance(k, tuple) else fmt % k
        v = str(v) if isinstance(v, tuple) else fmt % v
        k_len = max(k_len, len(k))
        items.append((k, v))
    if k_len > w - 5:
        raise ValueError("Key representation exceeds max len")
    v_len = w - 2 - k_len
    empty_k = k_len * ' ' + '  '
    lines = []
    for k, v in items:
        lines.append(': '.join((k.ljust(k_len), v[:v_len])))
        if len(v) >= v_len:
            for i in range(v_len, len(v) + v_len - 1, v_len):
                lines.append(empty_k + v[i: i + v_len])
                if i > v_len * max_v_lines:
                    lines.append(empty_k + '... ')
                    break
    lines[0] = '{' + lines[0]
    lines[-1] = lines[-1] + '}'
    return ',\n '.join(lines)


def printlist(list_obj, fmt='%r'):
    """Print each element of a list on a new line"""
    print(strlist(list_obj, fmt=fmt))


def repr_1line(obj, w=70):
    r = repr(obj)
    if len(r) > w:
        if isinstance(obj, np.ndarray):
            r = "<array, shape = %s>" % repr(obj.shape)
        elif isinstance(obj, (list, tuple)):
            suffix = '...' + r[-1] + ', len=%i' % len(obj)
            parts = r.split(',')
            maxlen = w - len(suffix)
            if len(parts[0]) > maxlen:
                r = parts[0][:maxlen] + suffix
            else:
                i = 1
                while sum(map(len, parts[:i + 1])) < maxlen and i < len(parts):
                    i += 1
                r = ','.join(parts[:i]) + suffix
        else:
            t = type(obj)
            try:
                l = len(obj)
                name = t.__name__
                r = '<%s, len = %i>' % (name, l)
            except BaseException:
                r = str(t)
    return r


def strlist(list_obj, fmt='%r'):
    lines = []
    for item in list_obj:
        rep = fmt % (item,)
        lines.append(rep)

    lines[0] = '[' + lines[0]
    lines[-1] = lines[-1] + ']'
    return ',\n '.join(lines)
