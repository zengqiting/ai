# 调试记录

## 问题1：OpenMP库冲突

**现象**：OMP: Error #15

**原因**：多个OpenMP运行时库冲突

**解决**：代码开头添加
```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
