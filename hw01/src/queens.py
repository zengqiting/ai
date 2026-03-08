def solve_n_queens(n):
    """解决N皇后问题，返回所有解"""
    def is_safe(board, row, col):
        # 检查同一列
        for i in range(row):
            if board[i] == col:
                return False
            # 故意引入的bug：对角线判断条件写错了
            # 正确的应该是 abs(board[i] - col) == abs(i - row)
            if abs(board[i] - col) == row - i:  # 这里有问题！
                return False
        return True
    
    def backtrack(board, row):
        if row == n:
            # 找到一个解
            result.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                backtrack(board, row + 1)
                board[row] = -1  # 回溯
    
    result = []
    board = [-1] * n
    backtrack(board, 0)
    return result

def print_solution(solution):
    """打印棋盘"""
    n = len(solution)
    for row in range(n):
        line = ['Q' if col == solution[row] else '.' for col in range(n)]
        print(' '.join(line))
    print()

def format_solution(solution):
    """格式化输出解（用于测试验证）"""
    n = len(solution)
    board = []
    for row in range(n):
        line = ['Q' if col == solution[row] else '.' for col in range(n)]
        board.append(''.join(line))
    return board
