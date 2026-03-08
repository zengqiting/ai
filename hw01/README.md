# 八皇后问题求解器

## 实现思路
使用回溯算法解决N皇后问题。核心思想是逐行放置皇后，每次放置时检查是否与已放置的皇后冲突（同列、对角线）。如果当前行找不到合适位置，则回溯到上一行。

## 文件结构
```
hw01/
├── src/
│   ├── __init__.py
│   └── queens.py      # 主逻辑代码
├── tests/
│   ├── __init__.py
│   └── test_queens.py # 单元测试
├── README.md          # 本文档
└── prompt_log.md      # AI交互日志
```

## 运行测试
在hw01目录下执行：
```bash
python -m unittest discover tests
```
或使用pytest：
```bash
pytest tests/ -v
```

## 使用示例
```python
from src.queens import solve_n_queens, print_solution

# 解决4皇后问题
solutions = solve_n_queens(4)
print(f"4皇后共有{len(solutions)}个解")
print_solution(solutions[0])

# 解决8皇后问题
solutions = solve_n_queens(8)
print(f"8皇后共有{len(solutions)}个解")
print_solution(solutions[0])
```

## 算法复杂度
- 时间复杂度：O(N!)
- 空间复杂度：O(N)
