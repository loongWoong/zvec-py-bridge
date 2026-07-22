"""测试 Python 模块 — 验证代码切分器。

这个模块包含几个函数和类，用于测试 chunker 的代码切分能力。
"""

import os
import sys
from typing import Optional, List


def add(a: int, b: int) -> int:
    """简单的加法函数。"""
    return a + b


def multiply(x: float, y: float) -> float:
    """乘法函数，带有类型注解。"""
    result = x * y
    return result


class Calculator:
    """一个简单的计算器类。"""

    def __init__(self, initial_value: float = 0.0):
        """初始化计算器。

        Args:
            initial_value: 初始值
        """
        self.value = initial_value
        self.history: List[float] = []

    def add(self, n: float) -> float:
        """加上一个数。

        Args:
            n: 要加的数

        Returns:
            新的值
        """
        self.value += n
        self.history.append(self.value)
        return self.value

    def subtract(self, n: float) -> float:
        """减去一个数。"""
        self.value -= n
        self.history.append(self.value)
        return self.value

    def multiply(self, n: float) -> float:
        """乘以一个数。"""
        self.value *= n
        self.history.append(self.value)
        return self.value

    def divide(self, n: float) -> Optional[float]:
        """除以一个数。如果除数为零返回 None。"""
        if n == 0:
            return None
        self.value /= n
        self.history.append(self.value)
        return self.value

    def reset(self) -> None:
        """重置计算器。"""
        self.value = 0.0
        self.history = []

    def get_history(self) -> List[float]:
        """获取计算历史。"""
        return self.history.copy()


def compute_average(numbers: List[float]) -> float:
    """计算平均值。

    Args:
        numbers: 数字列表

    Returns:
        平均值
    """
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def main() -> None:
    """主函数入口。"""
    calc = Calculator()
    calc.add(10)
    calc.multiply(3)
    calc.subtract(5)
    calc.divide(2)
    print(f"Final value: {calc.value}")
    print(f"History: {calc.get_history()}")


if __name__ == "__main__":
    main()
