from typing import Dict
from argparse import Action                

def _pair(value):
    if isinstance(value, tuple):
        if len(value) == 1:
            value = value * 2
        return value
    elif isinstance(value, list):
        return tuple(value)
    else:
        return (value, value)
    
def merge_parsers(exp, args):
    for arg_name in vars(args):
        arg_value = getattr(args, arg_name, None)
        if hasattr(exp, arg_name):
            assert not isinstance(arg_value, Dict), \
            "Unsupport Dict Type Set"
            check_parsers(exp, arg_name, arg_value, mode="tuple")
        else:
            check_parsers(exp, arg_name, arg_value, mode="dict")

def check_parsers(exp, name, value, mode="dict"):
    if mode == "dict":
        for exp_name, exp_value in vars(exp).items():
            if isinstance(exp_value, Dict):
                for n in exp_value:
                    if name == n:
                        exp_value[n] = value
                        setattr(exp, exp_name, exp_value)
                        return 0
    elif mode == "tuple":
        if isinstance(getattr(exp, name, None), list) or \
            isinstance(getattr(exp, name, None), tuple):
            setattr(exp, name, _pair(value))
        else:
            setattr(exp, name, value)
    else:
        raise NotImplemented
    return 0

                    
class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple: The expanded list or tuple from the string.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip('\'\"').replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        # for kv in values:
        #     key, val = kv.split('=', maxsplit=1)
        #     options[key] = self._parse_iterable(val)
        # setattr(namespace, self.dest, options)                     
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            setattr(namespace, key, self._parse_iterable(val))

        