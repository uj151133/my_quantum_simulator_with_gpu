# from models import node
# import math

import ctypes
ctypes.CDLL('/usr/local/lib/libginac.dylib')
# import calculation
import pybind11
import bit
import gate

# from sympy import symbols, sin, cos
# from sympy2ginac import sympy_to_ginac

# # シンボリック変数の定義
# x, y = symbols('x y')

# # 数式の定義
# expr1 = sin(x) * cos(y)
# expr2 = sin(x) * sin(y)

# # sympyの式をGiNaCの式に変換
# ginac_expr1 = sympy_to_ginac(expr1)
# ginac_expr2 = sympy_to_ginac(expr2)

# # 共通因数を計算
# common_factor_result = common_factor.common_factor(ginac_expr1, ginac_expr2)
# print("共通因数:", common_factor_result)
print(gate.X_GATE)