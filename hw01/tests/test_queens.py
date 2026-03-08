import unittest
from src.queens import solve_n_queens, format_solution

class TestQueens(unittest.TestCase):
    
    def test_4_queens_count(self):
        """测试4皇后问题应该有2个解"""
        solutions = solve_n_queens(4)
        self.assertEqual(len(solutions), 2, "4皇后应该有2个解")
    
    def test_8_queens_count(self):
        """测试8皇后问题应该有92个解"""
        solutions = solve_n_queens(8)
        self.assertEqual(len(solutions), 92, "8皇后应该有92个解")
    
    def test_4_queens_validity(self):
        """测试4皇后所有解的有效性"""
        solutions = solve_n_queens(4)
        for sol in solutions:
            self.assertTrue(self.is_valid_solution(sol), f"无效的解: {sol}")
    
    def test_8_queens_validity(self):
        """测试8皇后所有解的有效性"""
        solutions = solve_n_queens(8)
        for sol in solutions:
            self.assertTrue(self.is_valid_solution(sol), f"无效的解: {sol}")
    
    def is_valid_solution(self, board):
        """验证解是否有效（没有皇后互相攻击）"""
        n = len(board)
        for i in range(n):
            for j in range(i + 1, n):
                # 检查同一列
                if board[i] == board[j]:
                    return False
                # 检查对角线
                if abs(board[i] - board[j]) == abs(i - j):
                    return False
        return True
    
    def test_solution_format(self):
        """测试解的格式是否正确"""
        solutions = solve_n_queens(4)
        if solutions:
            formatted = format_solution(solutions[0])
            self.assertEqual(len(formatted), 4)
            for row in formatted:
                self.assertEqual(len(row), 4)
                self.assertTrue(all(c in 'Q.' for c in row))

if __name__ == '__main__':
    unittest.main()
